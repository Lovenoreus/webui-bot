#!/usr/bin/env python3
"""
Test script for Vanna SQL MCP tool
Tests the query_vanna_database endpoint with predefined Q&A pairs
"""

import json
import asyncio
import aiohttp
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import glob
load_dotenv(find_dotenv())


# Configuration
MCP_SERVER_URL = "http://localhost:8009"
QNA_FILE = "qna_for_vanna_sql.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class VannaTester:
    def __init__(self, mcp_server_url: str, qna_file: str, openai_api_key: str):
        self.mcp_server_url = mcp_server_url
        self.qna_file = qna_file
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.results = []
        self.test_timestamp = datetime.now()
        self.csv_file = f"vanna_test_results_{self.test_timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        self.results_dir = "test_results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test questions and answers from JSON file"""
        try:
            with open(self.qna_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('databases', [])
        except FileNotFoundError:
            print(f"Error: {self.qna_file} not found!")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
    
    async def query_vanna(self, question: str, keywords: List[str] = None) -> Dict[str, Any]:
        """Send question to Vanna MCP tool"""
        if keywords is None:
            keywords = []
            
        payload = {
            "query": question,
            "keywords": keywords
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/query_vanna_database",
                    json=payload,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        return {
                            "success": False,
                            "error": f"HTTP {response.status}: {await response.text()}"
                        }
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }
    
    def extract_answer_from_result(self, result: Dict[str, Any]) -> str:
        """Extract the answer from Vanna result"""
        if not result.get("success"):
            return f"ERROR: {result.get('error', 'Unknown error')}"
        
        results = result.get("results", [])
        if not results:
            return "No results returned"
        
        # Handle different result types
        if len(results) == 1:
            # Single result - extract the value
            first_result = results[0]
            if len(first_result) == 1:
                # Single column result
                return str(list(first_result.values())[0])
            else:
                # Multiple columns - format as key-value pairs
                return str(first_result)
        else:
            # Multiple results - return as list or count
            if len(results) <= 5:
                # Small result set - return the actual values
                return str(results)
            else:
                # Large result set - return count and sample
                return f"Count: {len(results)}, Sample: {results[:3]}"
    
    def grade_answer_with_openai(self, question: str, predicted_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Use OpenAI to grade the predicted answer against the correct answer"""
        if not self.openai_client:
            return {
                "correct": predicted_answer.strip() == correct_answer.strip(),
                "score": 1.0 if predicted_answer.strip() == correct_answer.strip() else 0.0,
                "reasoning": "OpenAI API key not available - using simple string comparison"
            }
        
        grading_prompt = f"""
You are an expert at grading SQL query results. Your task is to determine if the predicted answer matches the correct answer for a given question.

Question: {question}

Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}

Instructions:
1. Compare the predicted answer with the correct answer
2. Consider that:
   - Numeric answers can be represented differently (e.g., "3" vs "3.0" vs 3)
   - List answers can be in different formats but contain the same elements
   - Date formats may vary
   - Case sensitivity should be ignored for text
   - Whitespace differences should be ignored
   - Null/None values should be treated as equivalent

3. Return your assessment as JSON with these fields:
   - "correct": boolean (true if answers are equivalent, false otherwise)
   - "score": float (0.0 to 1.0, where 1.0 is perfect match)
   - "reasoning": string (brief explanation of your decision)

Examples:
- "3" and "3" → correct: true, score: 1.0
- "3" and "3.0" → correct: true, score: 1.0  
- "[1,2,3]" and "[1, 2, 3]" → correct: true, score: 1.0
- "John" and "john" → correct: true, score: 1.0
- "2023-11-05" and "2023-11-05T00:00:00" → correct: true, score: 1.0

Return only the JSON response:
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at grading SQL query results. Always respond with valid JSON only."},
                    {"role": "user", "content": grading_prompt}
                ],
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "correct": predicted_answer.strip().lower() == correct_answer.strip().lower(),
                    "score": 1.0 if predicted_answer.strip().lower() == correct_answer.strip().lower() else 0.0,
                    "reasoning": f"Failed to parse OpenAI response: {result_text}"
                }
                
        except Exception as e:
            return {
                "correct": predicted_answer.strip().lower() == correct_answer.strip().lower(),
                "score": 1.0 if predicted_answer.strip().lower() == correct_answer.strip().lower() else 0.0,
                "reasoning": f"OpenAI API error: {str(e)}"
            }
    
    def save_results_to_csv(self) -> None:
        """Save test results to CSV file"""
        csv_path = os.path.join(self.results_dir, self.csv_file)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'test_timestamp', 'database', 'question', 'correct_answer', 
                'predicted_answer', 'is_correct', 'score', 'reasoning',
                'sql_query', 'record_count', 'method'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow({
                    'test_timestamp': result['timestamp'],
                    'database': result['database'],
                    'question': result['question'],
                    'correct_answer': result['correct_answer'],
                    'predicted_answer': result['predicted_answer'],
                    'is_correct': result['grading']['correct'],
                    'score': result['grading']['score'],
                    'reasoning': result['grading']['reasoning'],
                    'sql_query': result['vanna_result'].get('sql_query', ''),
                    'record_count': result['vanna_result'].get('record_count', 0),
                    'method': result['vanna_result'].get('method', 'vanna')
                })
        
        print(f"💾 CSV results saved to: {csv_path}")
    
    def load_previous_results(self) -> pd.DataFrame:
        """Load all previous test results from CSV files"""
        csv_files = glob.glob(os.path.join(self.results_dir, "vanna_test_results_*.csv"))
        
        if not csv_files:
            return pd.DataFrame()
        
        all_results = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_results.append(df)
            except Exception as e:
                print(f"⚠️  Warning: Could not load {csv_file}: {e}")
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def compare_with_previous(self, previous_df: pd.DataFrame) -> None:
        """Compare current results with previous runs"""
        if previous_df.empty:
            print("📊 No previous results found for comparison")
            return
        
        current_accuracy = sum(1 for r in self.results if r['grading']['correct']) / len(self.results)
        current_avg_score = sum(r['grading']['score'] for r in self.results) / len(self.results)
        
        # Get latest previous run
        latest_previous = previous_df[previous_df['test_timestamp'] != self.test_timestamp.isoformat()]
        if not latest_previous.empty:
            latest_timestamp = latest_previous['test_timestamp'].max()
            latest_run = latest_previous[latest_previous['test_timestamp'] == latest_timestamp]
            
            prev_accuracy = latest_run['is_correct'].mean()
            prev_avg_score = latest_run['score'].mean()
            
            print(f"\n📈 COMPARISON WITH PREVIOUS RUN ({latest_timestamp})")
            print("-" * 60)
            print(f"Accuracy: {current_accuracy:.1%} (Previous: {prev_accuracy:.1%}) - {self.get_change_indicator(current_accuracy, prev_accuracy)}")
            print(f"Avg Score: {current_avg_score:.2f} (Previous: {prev_avg_score:.2f}) - {self.get_change_indicator(current_avg_score, prev_avg_score)}")
            
            # Question-by-question comparison
            print(f"\n📝 QUESTION-BY-QUESTION COMPARISON:")
            for result in self.results:
                question = result['question'][:50] + "..." if len(result['question']) > 50 else result['question']
                current_correct = result['grading']['correct']
                
                # Find matching question in previous run
                matching_questions = latest_run[latest_run['question'] == result['question']]
                if not matching_questions.empty:
                    prev_correct = matching_questions.iloc[0]['is_correct']
                    status = "✅" if current_correct else "❌"
                    change = ""
                    if current_correct != prev_correct:
                        change = f" (Changed: {'❌→✅' if current_correct else '✅→❌'})"
                    print(f"  {status} {question}{change}")
                else:
                    status = "✅" if current_correct else "❌"
                    print(f"  {status} {question} (New question)")
    
    def get_change_indicator(self, current: float, previous: float) -> str:
        """Get indicator for change direction"""
        diff = current - previous
        if abs(diff) < 0.01:
            return "➡️ Same"
        elif diff > 0:
            return f"📈 +{diff:.1%}"
        else:
            return f"📉 {diff:.1%}"
    
    def plot_results_history(self) -> None:
        """Plot graph of all test runs"""
        previous_df = self.load_previous_results()
        
        if previous_df.empty:
            print("📊 No previous results to plot")
            return
        
        # Prepare data for plotting
        current_run_data = pd.DataFrame([{
            'test_timestamp': self.test_timestamp.isoformat(),
            'accuracy': sum(1 for r in self.results if r['grading']['correct']) / len(self.results),
            'avg_score': sum(r['grading']['score'] for r in self.results) / len(self.results),
            'total_questions': len(self.results),
            'correct_answers': sum(1 for r in self.results if r['grading']['correct'])
        }])
        
        # Combine with previous results
        all_runs = []
        
        # Group previous results by timestamp
        for timestamp in previous_df['test_timestamp'].unique():
            run_data = previous_df[previous_df['test_timestamp'] == timestamp]
            all_runs.append({
                'test_timestamp': timestamp,
                'accuracy': run_data['is_correct'].mean(),
                'avg_score': run_data['score'].mean(),
                'total_questions': len(run_data),
                'correct_answers': run_data['is_correct'].sum()
            })
        
        # Add current run
        all_runs.append({
            'test_timestamp': current_run_data.iloc[0]['test_timestamp'],
            'accuracy': current_run_data.iloc[0]['accuracy'],
            'avg_score': current_run_data.iloc[0]['avg_score'],
            'total_questions': current_run_data.iloc[0]['total_questions'],
            'correct_answers': current_run_data.iloc[0]['correct_answers']
        })
        
        df_plot = pd.DataFrame(all_runs)
        df_plot['test_timestamp'] = pd.to_datetime(df_plot['test_timestamp'])
        df_plot = df_plot.sort_values('test_timestamp')
        df_plot['run_number'] = range(1, len(df_plot) + 1)
        
        # Create the plot
        plt.style.use('seaborn-v0_8')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Vanna SQL Test Results History', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy over time
        ax1.plot(df_plot['run_number'], df_plot['accuracy'] * 100, 'b-o', linewidth=2, markersize=6)
        ax1.set_title('Accuracy Over Time', fontweight='bold')
        ax1.set_xlabel('Test Run Number')
        ax1.set_ylabel('Accuracy (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Plot 2: Average Score over time
        ax2.plot(df_plot['run_number'], df_plot['avg_score'], 'g-o', linewidth=2, markersize=6)
        ax2.set_title('Average Score Over Time', fontweight='bold')
        ax2.set_xlabel('Test Run Number')
        ax2.set_ylabel('Average Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Total Questions vs Correct Answers
        ax3.bar(df_plot['run_number'] - 0.2, df_plot['total_questions'], 0.4, label='Total Questions', alpha=0.7)
        ax3.bar(df_plot['run_number'] + 0.2, df_plot['correct_answers'], 0.4, label='Correct Answers', alpha=0.7)
        ax3.set_title('Questions vs Correct Answers', fontweight='bold')
        ax3.set_xlabel('Test Run Number')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Accuracy distribution (box plot if multiple runs)
        if len(df_plot) > 1:
            ax4.boxplot([df_plot['accuracy'] * 100], labels=['Accuracy'])
            ax4.set_title('Accuracy Distribution', fontweight='bold')
            ax4.set_ylabel('Accuracy (%)')
        else:
            ax4.text(0.5, 0.5, 'Multiple runs needed\nfor distribution plot', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Accuracy Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(self.results_dir, f"vanna_test_history_{self.test_timestamp.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {plot_file}")
        
        # Also save a summary plot
        self.save_summary_plot(df_plot)
        
        plt.show()
    
    def save_summary_plot(self, df_plot: pd.DataFrame) -> None:
        """Save a summary plot with key metrics"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Create a comprehensive summary plot
        ax2 = ax.twinx()
        
        # Plot accuracy and score
        line1 = ax.plot(df_plot['run_number'], df_plot['accuracy'] * 100, 'b-o', 
                       linewidth=2, markersize=6, label='Accuracy (%)')
        line2 = ax2.plot(df_plot['run_number'], df_plot['avg_score'], 'g-s', 
                        linewidth=2, markersize=6, label='Avg Score')
        
        # Styling
        ax.set_xlabel('Test Run Number', fontweight='bold')
        ax.set_ylabel('Accuracy (%)', color='b', fontweight='bold')
        ax2.set_ylabel('Average Score', color='g', fontweight='bold')
        ax.set_title('Vanna SQL Test Performance Summary', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        
        # Grid and limits
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_plot = os.path.join(self.results_dir, "vanna_test_summary.png")
        plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
        print(f"📈 Summary plot saved to: {summary_plot}")
        plt.close()
    
    async def run_tests(self) -> None:
        """Run all tests and generate report"""
        databases = self.load_test_data()
        
        if not databases:
            print("No test data found!")
            return
        
        print(f"🧪 Starting Vanna SQL Test Suite")
        print(f"📅 Test Date: {self.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔗 MCP Server: {self.mcp_server_url}")
        print(f"📊 Databases to test: {len(databases)}")
        print(f"📁 Results will be saved to: {self.results_dir}/")
        print("=" * 80)
        
        total_questions = 0
        total_correct = 0
        total_score = 0.0
        
        for db_idx, database in enumerate(databases, 1):
            db_name = database.get('database_name', 'Unknown')
            db_location = database.get('database_location', 'Unknown')
            db_type = database.get('database_type', 'Unknown')
            questions = database.get('questions', [])
            
            print(f"\n📁 Database {db_idx}: {db_name}")
            print(f"   Location: {db_location}")
            print(f"   Type: {db_type}")
            print(f"   Questions: {len(questions)}")
            print("-" * 60)
            
            for q_idx, qa_pair in enumerate(questions, 1):
                question = qa_pair.get('question', '')
                correct_answer = qa_pair.get('answer', '')
                
                print(f"\n❓ Question {q_idx}: {question}")
                
                # Query Vanna
                result = await self.query_vanna(question)
                predicted_answer = self.extract_answer_from_result(result)
                
                print(f"🤖 Vanna Answer: {predicted_answer}")
                print(f"✅ Correct Answer: {correct_answer}")
                
                # Grade the answer
                grading = self.grade_answer_with_openai(question, predicted_answer, correct_answer)
                
                print(f"📊 Grade: {'✅ CORRECT' if grading['correct'] else '❌ INCORRECT'}")
                print(f"🎯 Score: {grading['score']:.2f}")
                print(f"💭 Reasoning: {grading['reasoning']}")
                
                # Store results
                test_result = {
                    "database": db_name,
                    "question": question,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "vanna_result": result,
                    "grading": grading,
                    "timestamp": datetime.now().isoformat()
                }
                self.results.append(test_result)
                
                # Update totals
                total_questions += 1
                if grading['correct']:
                    total_correct += 1
                total_score += grading['score']
        
        # Generate final report
        print("\n" + "=" * 80)
        print("📈 FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Total Questions: {total_questions}")
        print(f"Correct Answers: {total_correct}")
        print(f"Accuracy: {(total_correct/total_questions)*100:.1f}%" if total_questions > 0 else "N/A")
        print(f"Average Score: {(total_score/total_questions):.2f}" if total_questions > 0 else "N/A")
        
        # Save results to CSV
        self.save_results_to_csv()
        
        # Load and compare with previous results
        previous_df = self.load_previous_results()
        self.compare_with_previous(previous_df)
        
        # Generate plots
        print(f"\n📊 Generating plots...")
        self.plot_results_history()
        
        # Save detailed JSON results
        results_file = os.path.join(self.results_dir, f"vanna_test_results_{self.test_timestamp.strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_summary": {
                    "total_questions": total_questions,
                    "correct_answers": total_correct,
                    "accuracy_percentage": (total_correct/total_questions)*100 if total_questions > 0 else 0,
                    "average_score": total_score/total_questions if total_questions > 0 else 0,
                    "test_date": self.test_timestamp.isoformat(),
                    "mcp_server_url": self.mcp_server_url
                },
                "detailed_results": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 JSON results saved to: {results_file}")
        print(f"\n🎉 Test completed successfully!")

async def main():
    """Main function to run the tests"""
    if not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY not set. Will use simple string comparison for grading.")
    
    tester = VannaTester(MCP_SERVER_URL, QNA_FILE, OPENAI_API_KEY)
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main())
