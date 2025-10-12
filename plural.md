# Singular/Plural Query Processing Flow

This document describes the complete flow for handling singular/plural queries in the SQL generation system, ensuring comprehensive search results regardless of how items are stored in the database.

## Overview

The system dynamically converts between singular and plural forms of words in both Swedish and English, creating SQL queries with OR conditions to search for all possible variants. This ensures that a query for "Swimming trunks" will find items stored as both "Swimming trunk" and "Swimming trunks".

## Complete Processing Flow

### 1. Query Input
User submits a natural language query:
```
"What companies have sold us Swimming trunks?"
```

### 2. Query Preprocessing (server_new.py)
The main server imports and calls the normalization function:
```python
from training import normalize_for_comprehensive_search

# The query gets preprocessed to enhance search coverage
enhanced_query = normalize_for_comprehensive_search(query)
```

### 3. Dynamic Analysis (training.py)
The `normalize_for_comprehensive_search()` function performs:

1. **Word Tokenization**: Splits the query into individual words
2. **Stop Word Filtering**: Removes common words (what, companies, have, sold, us, etc.)
3. **Item Term Identification**: Identifies potential item names ("swimming", "trunks")
4. **Variant Generation**: For each item term, calls `generate_all_singular_plural_variants()`

### 4. Dynamic Variant Generation
```python
generate_all_singular_plural_variants("trunks")
```

This core function:
1. **Language Detection**: Uses `is_swedish_word()` to determine Swedish vs English
2. **Singular Conversion**: Uses `pluralize_to_singular()` with language-specific rules
3. **Plural Generation**: Uses `singular_to_plural()` with language-specific rules
4. **Variant Collection**: Returns all legitimate forms as a set

**Example Processing:**
- Input: `"trunks"`
- Language Detection: English (no Swedish characters/patterns)
- Singular Form: `"trunk"` (removes 's')
- Plural Form: `"trunks"` (adds 's' back)
- Output: `{"trunk", "trunks"}`

### 5. Enhanced Instructions Generation
The system creates comprehensive search instructions:
```
DYNAMIC SINGULAR/PLURAL SEARCH INSTRUCTIONS: 
When searching for item names, use OR conditions to search for ALL singular and plural forms to get complete results.

For key item terms in the query, generate LIKE conditions for all forms:
'trunks' -> search all forms: trunk, trunks

SQL Pattern Examples:
- WHERE (LOWER(ITEM_NAME) LIKE LOWER('%swimming trunk%') OR LOWER(ITEM_NAME) LIKE LOWER('%swimming trunks%'))
```

### 6. Vanna AI Processing
The enhanced query with instructions is sent to Vanna AI, which generates SQL with OR conditions:
```sql
SELECT DISTINCT i.SUPPLIER_PARTY_NAME
FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
    ON i.INVOICE_ID = il.INVOICE_ID
WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunk%') 
    OR LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunks%'))
```

### 7. Database Query Execution
The comprehensive SQL finds items stored in any variant:
- "Swimming Trunk Size L"
- "Swimming Trunks Blue"
- "Men's Swimming Trunks"
- "Water Swimming Trunk"

## Key Components Architecture

### Language Detection Engine
```python
is_swedish_word("skruvar")  # Returns True (Swedish 'ar' ending)
is_swedish_word("screws")   # Returns False (English pattern)
```

**Detection Criteria:**
- Swedish characters: å, ä, ö
- Swedish patterns: sj, skj, tj, kj
- Swedish endings: -ar, -or, -er, -ning, -het
- Known Swedish words in dictionary

### Dynamic Pluralization Engine

**English Rules:**
```python
pluralize_to_singular("screws") → "screw"      # Remove 's'
pluralize_to_singular("batteries") → "battery"  # 'ies' → 'y'
pluralize_to_singular("knives") → "knife"      # 'ves' → 'f'

singular_to_plural("screw") → "screws"         # Add 's'
singular_to_plural("battery") → "batteries"    # 'y' → 'ies'
singular_to_plural("knife") → "knives"         # 'f' → 'ves'
```

**Swedish Rules:**
```python
pluralize_to_singular("skruvar") → "skruv"     # Remove 'ar'
pluralize_to_singular("flickor") → "flicka"    # 'or' → 'a'
pluralize_to_singular("datorer") → "dator"     # 'orer' → 'or'

singular_to_plural("skruv") → "skruvar"        # Add 'ar'
singular_to_plural("flicka") → "flickor"       # 'a' → 'or'
singular_to_plural("dator") → "datorer"        # 'or' → 'orer'
```

### Complete Variant Generation
```python
generate_all_singular_plural_variants("batteries")
# Returns: {"battery", "batteries"}

generate_all_singular_plural_variants("skruvar") 
# Returns: {"skruv", "skruvar"}

generate_all_singular_plural_variants("teeth")
# Returns: {"tooth", "teeth"}  # Uses irregular forms
```

## Flow Diagram

```mermaid
graph TD
    A[User Query: "What companies sold us swimming trunks?"] --> B[normalize_for_comprehensive_search]
    B --> C[Split into words & filter stop words]
    C --> D[Identify item terms: swimming, trunks]
    D --> E[For each term: generate_all_singular_plural_variants]
    E --> F[trunks → {trunk, trunks}]
    F --> G[Create enhanced instructions with OR conditions]
    G --> H[Send to Vanna AI with training context]
    H --> I[Vanna generates SQL with OR conditions]
    I --> J[Database returns comprehensive results]
```

## Language-Specific Examples

### English Processing
```
Input: "Show me all screws"
↓
Term: "screws"
↓
Language: English (detected)
↓
Variants: {"screw", "screws"}
↓
SQL: WHERE (LOWER(ITEM_NAME) LIKE '%screw%' OR LOWER(ITEM_NAME) LIKE '%screws%')
```

### Swedish Processing
```
Input: "Visa alla skruvar"
↓
Term: "skruvar"
↓
Language: Swedish (detected by 'ar' ending)
↓
Variants: {"skruv", "skruvar"}
↓
SQL: WHERE (LOWER(ITEM_NAME) LIKE '%skruv%' OR LOWER(ITEM_NAME) LIKE '%skruvar%')
```

### Irregular Forms
```
Input: "dental teeth supplies"
↓
Term: "teeth"
↓
Language: English
↓
Irregular Form Detected: teeth ↔ tooth
↓
Variants: {"teeth", "tooth"}
↓
SQL: WHERE (LOWER(ITEM_NAME) LIKE '%teeth%' OR LOWER(ITEM_NAME) LIKE '%tooth%')
```

## System Benefits

### 1. Fully Dynamic Operation
- ✅ **No Static Lists**: Handles any English/Swedish word using linguistic rules
- ✅ **Future-Proof**: New words work automatically without system updates
- ✅ **Scalable**: No maintenance overhead for vocabulary expansion

### 2. Linguistic Accuracy
- ✅ **Proper Grammar Rules**: Uses real linguistic patterns, not simple heuristics
- ✅ **Language-Aware**: Applies correct Swedish vs English pluralization
- ✅ **Irregular Forms**: Handles exceptions like teeth/tooth, katt/katter

### 3. Comprehensive Coverage
- ✅ **Complete Results**: Finds items regardless of storage form (singular/plural)
- ✅ **Flexible Input**: Users can ask in any form and get complete results
- ✅ **Data Consistency**: Overcomes database inconsistencies in item naming

### 4. Clean Architecture
- ✅ **Separation of Concerns**: Logic isolated in training.py module
- ✅ **Testable Components**: Each function has clear, testable responsibility
- ✅ **Maintainable Code**: Well-documented, modular design

## Error Handling & Edge Cases

### Invalid Input
- Empty strings: Returns original input
- Very short words (< 2 chars): Returns original input
- Non-alphabetic characters: Filtered out during processing

### Language Ambiguity
- Borderline cases: System tries both Swedish and English rules
- Unknown patterns: Falls back to basic 's' addition/removal
- Mixed languages: Each word processed independently

### Database Coverage
- Missing variants: OR conditions ensure maximum coverage
- Inconsistent naming: System finds items regardless of storage format
- Special characters: LOWER() function handles case insensitivity

## Performance Considerations

- **Minimal Overhead**: Fast linguistic rule application
- **Efficient SQL**: OR conditions are database-optimized
- **Caching Potential**: Variants could be cached for repeated queries
- **Scalable Design**: No exponential complexity with vocabulary growth

## Future Enhancements

1. **Extended Language Support**: Add Norwegian, Danish, German
2. **Compound Word Handling**: Better support for Swedish compound words
3. **Semantic Variants**: Include synonyms and related terms
4. **Context Awareness**: Use surrounding words to improve variant selection
5. **Machine Learning**: Train models on actual database contents for optimization

---

*This system ensures that users get comprehensive search results regardless of how they phrase their queries or how items are stored in the database, providing a robust foundation for natural language SQL generation.*