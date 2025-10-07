# Vanna SQL Test Suite

## 🧪 Enhanced Testing Features

The Vanna SQL test suite has been enhanced with comprehensive testing, analysis, and visualization capabilities.

### 📊 New Features

1. **CSV Export**: All test results are automatically saved to timestamped CSV files
2. **Result Comparison**: Compares current run with previous test results
3. **Performance Plotting**: Generates graphs showing test performance over time
4. **Historical Analysis**: Tracks accuracy and score trends across multiple runs

### 🚀 Usage

```bash
# Run the test suite
./run_vanna_tests.sh

# Or run directly
python test_vanna_sql.py
```

### 📁 Output Files

Each test run creates files in the `test_results/` directory:

- **CSV File**: `vanna_test_results_YYYYMMDD_HHMMSS.csv`
- **JSON File**: `vanna_test_results_YYYYMMDD_HHMMSS.json`
- **History Plot**: `vanna_test_history_YYYYMMDD_HHMMSS.png`
- **Summary Plot**: `vanna_test_summary.png` (updated with each run)

### 📈 CSV Format

The CSV includes these columns:
- `test_timestamp`: When the test was run
- `database`: Database being tested
- `question`: Test question
- `correct_answer`: Expected answer
- `predicted_answer`: Vanna's answer
- `is_correct`: Boolean result
- `score`: Grade score (0.0-1.0)
- `reasoning`: AI grading explanation
- `sql_query`: Generated SQL query
- `record_count`: Number of records returned
- `method`: Testing method used

### 📊 Plot Features

The generated plots show:
1. **Accuracy Over Time**: Percentage of correct answers per run
2. **Average Score Over Time**: Average grading score per run
3. **Questions vs Correct Answers**: Bar chart comparison
4. **Accuracy Distribution**: Box plot of accuracy ranges

### 🔄 Comparison Features

- **Run-to-Run Comparison**: Shows changes from previous test
- **Question-by-Question Analysis**: Tracks which questions improved/declined
- **Trend Indicators**: Visual indicators for performance changes (📈📉➡️)

### 📋 Sample Output

```
📈 COMPARISON WITH PREVIOUS RUN (2024-12-19T14:30:25)
------------------------------------------------------------
Accuracy: 83.3% (Previous: 75.0%) - 📈 +8.3%
Avg Score: 0.92 (Previous: 0.85) - 📈 +0.07

📝 QUESTION-BY-QUESTION COMPARISON:
  ✅ Hur många fakturor har finns det...
  ✅ Hur mycket pengar lade vi på hotell...
  ❌ Hur många fakturor har vi som betalades... (Changed: ✅→❌)
```

### 🎯 Benefits

- **Performance Tracking**: Monitor Vanna's accuracy improvements over time
- **Regression Detection**: Quickly identify when performance degrades
- **Question Analysis**: Understand which types of questions are challenging
- **Historical Data**: Maintain complete test history for analysis
- **Visual Insights**: Easy-to-read graphs for stakeholders

### 🔧 Configuration

Edit `qna_for_vanna_sql.json` to:
- Add new test questions
- Include additional databases
- Modify expected answers
- Add metadata for documentation

The test suite is designed to be extensible and maintainable for long-term testing of your Vanna SQL implementation!
