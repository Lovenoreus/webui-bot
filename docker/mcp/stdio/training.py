# -------------------- External Libraries --------------------
# from dotenv import load_dotenv, find_dotenv
import re

# # -------------------- User-defined Modules --------------------
# import config

# load_dotenv(find_dotenv())


def pluralize_to_singular(word):
    """
    Convert a single word from plural to singular using Swedish and English pluralization rules.
    
    Args:
        word (str): The word to convert
        
    Returns:
        str: The singular form of the word
    """
    if len(word) < 2:
        return word
    
    # Check if it's likely a Swedish word
    if is_swedish_word(word):
        return pluralize_to_singular_swedish(word)
    
    word_lower = word.lower()
    
    # Handle English irregular plurals
    irregular_plurals = {
        'children': 'child',
        'feet': 'foot',
        'geese': 'goose',
        'men': 'man',
        'women': 'woman',
        'teeth': 'tooth',
        'mice': 'mouse',
        'people': 'person',
        'oxen': 'ox',
        'deer': 'deer',
        'sheep': 'sheep',
        'fish': 'fish',
        'moose': 'moose',
        'series': 'series',
        'species': 'species',
        'data': 'datum',
        'media': 'medium',
        'criteria': 'criterion',
        'phenomena': 'phenomenon',
        'bacteria': 'bacterium',
        'alumni': 'alumnus',
        'fungi': 'fungus',
        'nuclei': 'nucleus',
        'cacti': 'cactus',
        'foci': 'focus',
        'radii': 'radius',
        'analyses': 'analysis',
        'bases': 'basis',
        'diagnoses': 'diagnosis',
        'oases': 'oasis',
        'theses': 'thesis',
        'crises': 'crisis',
        'axes': 'axis',
        'matrices': 'matrix',
        'vertices': 'vertex',
        'indices': 'index',
        'appendices': 'appendix'
    }
    
    if word_lower in irregular_plurals:
        # Preserve original case pattern
        singular = irregular_plurals[word_lower]
        if word.isupper():
            return singular.upper()
        elif word.istitle():
            return singular.capitalize()
        else:
            return singular
    
    # Handle regular English plural patterns
    
    # Words ending in 'ies' -> 'y' (e.g., companies -> company, batteries -> battery)
    if word_lower.endswith('ies') and len(word) > 3:
        base = word[:-3] + 'y'
        return base
    
    # Words ending in 'ves' -> 'f' or 'fe' (e.g., knives -> knife, shelves -> shelf)
    if word_lower.endswith('ves') and len(word) > 3:
        if word_lower.endswith('ives'):
            # knives -> knife, lives -> life
            base = word[:-4] + 'ife'
        else:
            # shelves -> shelf, calves -> calf
            base = word[:-3] + 'f'
        return base
    
    # Words ending in 'ses' -> 's' (e.g., glasses -> glass, classes -> class)
    if word_lower.endswith('ses') and len(word) > 3:
        # Special case for 'chases', 'purchases', 'releases', etc.
        if word_lower.endswith('chases') or word_lower.endswith('eases'):
            return word[:-1]  # Remove just the 's'
        return word[:-2]
    
    # Words ending in 'xes' -> 'x' (e.g., boxes -> box, fixes -> fix)
    if word_lower.endswith('xes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in 'zes' -> 'z' (e.g., prizes -> prize)
    if word_lower.endswith('zes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in 'shes' -> 'sh' (e.g., dishes -> dish, brushes -> brush)
    if word_lower.endswith('shes') and len(word) > 4:
        return word[:-2]
    
    # Words ending in 'ches' -> 'ch' (e.g., watches -> watch, beaches -> beach)
    if word_lower.endswith('ches') and len(word) > 4:
        return word[:-2]
    
    # Words ending in 'oes' -> 'o' (e.g., tomatoes -> tomato, heroes -> hero)
    if word_lower.endswith('oes') and len(word) > 3:
        return word[:-2]
    
    # Words ending in just 's' (most common case for English)
    if word_lower.endswith('s') and len(word) > 1:
        # Don't remove 's' from words that are naturally singular and end in 's'
        # (e.g., glass, pass, mass, etc.)
        potential_singular = word[:-1]
        
        # Simple heuristic: if removing 's' creates a very short word, it might be incorrect
        if len(potential_singular) < 2:
            return word
        
        # Special cases where the word naturally ends in 's' when singular
        if word_lower in ['glass', 'mass', 'pass', 'class', 'bass', 'grass', 'brass', 'cross']:
            return word
            
        # For most regular English plurals, just remove the 's'
        return potential_singular
    
    # If no plural pattern matches, return the original word
    return word


def is_swedish_word(word):
    """
    Detect if a word is likely Swedish based on common Swedish characteristics.
    """
    word_lower = word.lower()
    
    # Known English words (to avoid false positives)
    known_english = {
        'computer', 'computers', 'battery', 'batteries', 'screw', 'screws',
        'knife', 'knives', 'swimming', 'trunk', 'trunks', 'the', 'and', 'or',
        'what', 'which', 'how', 'many', 'show', 'find', 'all', 'purchases'
    }
    
    if word_lower in known_english:
        return False
    
    # Known Swedish words (common ones to help with detection)
    known_swedish = {
        'skruv', 'skruvar', 'bil', 'bilar', 'flicka', 'flickor', 'katt', 'katter', 
        'hus', 'pojke', 'pojkar', 'batteri', 'batterier', 'dator', 'datorer',
        'kniv', 'knivar', 'företag', 'vilka', 'sålde', 'köpte', 'alla', 'visa'
    }
    
    if word_lower in known_swedish:
        return True
    
    # Swedish-specific characters (definitive indicators)
    if any(char in word_lower for char in ['å', 'ä', 'ö']):
        return True
    
    # Swedish-specific letter combinations and patterns
    swedish_patterns = [
        'sj', 'skj', 'tj', 'kj',  # Swedish consonant combinations
    ]
    
    # Swedish common endings (but be careful with English words)
    swedish_endings = [
        'ning', 'het', 'dom', 'skap', 'else', 'are', 'ere', 'ande', 'ende'
    ]
    
    # Check for Swedish-specific patterns
    for pattern in swedish_patterns:
        if pattern in word_lower:
            return True
    
    # Check for Swedish-specific endings (but not if it's likely English)
    for ending in swedish_endings:
        if word_lower.endswith(ending) and len(word) > 4:
            return True
    
    # If word ends with typical Swedish plural patterns (but check length and context)
    if len(word) > 3:
        if (word_lower.endswith('ar') and not word_lower.endswith('lar')) or word_lower.endswith('or'):
            return True
        if word_lower.endswith('er') and len(word) > 5:  # Avoid short English words like "her"
            return True
    
    return False


def singular_to_plural_swedish(word):
    """
    Convert Swedish singular to plural using Swedish pluralization rules.
    """
    word_lower = word.lower()
    
    # Swedish irregular plurals
    swedish_irregulars = {
        'man': 'män',
        'kvinna': 'kvinnor',
        'barn': 'barn',  # Same form
        'mus': 'möss',
        'gås': 'gäss',
        'tand': 'tänder',
        'fot': 'fötter',
        'hand': 'händer',
        'öga': 'ögon',
        'öra': 'öron',
        'hus': 'hus',  # Same form
        'katt': 'katter'  # Double consonant case
    }
    
    if word_lower in swedish_irregulars:
        plural = swedish_irregulars[word_lower]
        if word.isupper():
            return plural.upper()
        elif word.istitle():
            return plural.capitalize()
        else:
            return plural
    
    # Swedish pluralization patterns
    
    # Words ending in 'a' -> remove 'a' and add 'or' (flicka -> flickor)
    if word_lower.endswith('a') and len(word) > 2:
        return word[:-1] + 'or'
    
    # Words ending in 'e' -> remove 'e' and add 'ar' (pojke -> pojkar)
    if word_lower.endswith('e') and len(word) > 2:
        return word[:-1] + 'ar'
    
    # Words ending in consonant
    if word_lower[-1] not in 'aeiouåäö' and len(word) > 1:
        # Special double consonant cases (katt -> katter)
        # Only for single consonants that typically double in Swedish
        if (word_lower[-1] in 'tmn' and len(word) > 2 and 
            word_lower[-2] in 'aeiouåäö'):  # Vowel before the consonant
            return word + word[-1] + 'er'
        else:
            # Regular consonant ending -> add 'ar' (bil -> bilar)
            return word + 'ar'
    
    # Words ending in vowels (not 'a' or 'e') -> add 'r' 
    return word + 'r'


def pluralize_to_singular_swedish(word):
    """
    Convert Swedish plural to singular using Swedish pluralization rules.
    """
    word_lower = word.lower()
    
    # Swedish irregular plurals (reverse mapping)
    swedish_irregular_plurals = {
        'män': 'man',
        'kvinnor': 'kvinna',
        'barn': 'barn',  # Same form
        'möss': 'mus',
        'gäss': 'gås',
        'tänder': 'tand',
        'fötter': 'fot',
        'händer': 'hand',
        'ögon': 'öga',
        'öron': 'öra',
        'hus': 'hus',  # Same form
        'katter': 'katt'  # Double consonant case
    }
    
    if word_lower in swedish_irregular_plurals:
        singular = swedish_irregular_plurals[word_lower]
        if word.isupper():
            return singular.upper()
        elif word.istitle():
            return singular.capitalize()
        else:
            return singular
    
    # Swedish singular patterns (reverse of plural rules)
    
    # Words ending in 'or' -> remove 'or' and add 'a' (flickor -> flicka)
    if word_lower.endswith('or') and len(word) > 3:
        return word[:-2] + 'a'
    
    # Words ending in 'er' -> handle double consonant (katter -> katt)
    if word_lower.endswith('er') and len(word) > 4:
        # Check for double consonant pattern (katter -> katt)
        if len(word) > 5 and word[-3] == word[-4]:
            return word[:-3]  # Remove one consonant + 'er'
        else:
            return word[:-2]  # Just remove 'er'
    
    # Words ending in 'ar' -> remove 'ar' (bilar -> bil, pojkar -> pojke)
    if word_lower.endswith('ar') and len(word) > 3:
        # Check if it should end in 'e' (pojkar -> pojke)
        base = word[:-2]
        # Simple heuristic: if base ends in consonant cluster, might need 'e'
        if len(base) > 2 and base[-1] in 'kgtp' and base[-2] in 'jln':
            return base + 'e'
        else:
            return base
    
    # Words ending in single 'r' -> remove 'r' (special cases)
    if (word_lower.endswith('r') and not word_lower.endswith(('ar', 'er', 'or')) 
        and len(word) > 2):
        return word[:-1]
    
    # If no pattern matches, return original
    return word


def singular_to_plural(word):
    """
    Convert a single word from singular to plural using Swedish and English pluralization rules.
    
    Args:
        word (str): The word to convert
        
    Returns:
        str: The plural form of the word
    """
    if len(word) < 2:
        return word
    
    # Check if it's likely a Swedish word
    if is_swedish_word(word):
        return singular_to_plural_swedish(word)
    
    word_lower = word.lower()
    
    # Handle English irregular plurals
    irregular_singulars = {
        'child': 'children',
        'foot': 'feet',
        'goose': 'geese',
        'man': 'men',
        'woman': 'women',
        'tooth': 'teeth',
        'mouse': 'mice',
        'person': 'people',
        'ox': 'oxen',
        'deer': 'deer',
        'sheep': 'sheep',
        'fish': 'fish',
        'moose': 'moose',
        'series': 'series',
        'species': 'species',
        'datum': 'data',
        'medium': 'media',
        'criterion': 'criteria',
        'phenomenon': 'phenomena',
        'bacterium': 'bacteria',
        'alumnus': 'alumni',
        'fungus': 'fungi',
        'nucleus': 'nuclei',
        'cactus': 'cacti',
        'focus': 'foci',
        'radius': 'radii',
        'analysis': 'analyses',
        'basis': 'bases',
        'diagnosis': 'diagnoses',
        'oasis': 'oases',
        'thesis': 'theses',
        'crisis': 'crises',
        'axis': 'axes',
        'matrix': 'matrices',
        'vertex': 'vertices',
        'index': 'indices',
        'appendix': 'appendices'
    }
    
    if word_lower in irregular_singulars:
        # Preserve original case pattern
        plural = irregular_singulars[word_lower]
        if word.isupper():
            return plural.upper()
        elif word.istitle():
            return plural.capitalize()
        else:
            return plural
    
    # Handle regular English singular to plural patterns
    
    # Words ending in 'y' preceded by a consonant -> 'ies' (e.g., company -> companies, battery -> batteries)
    if word_lower.endswith('y') and len(word) > 2 and word[-2].lower() not in 'aeiou':
        return word[:-1] + 'ies'
    
    # Words ending in 'f' or 'fe' -> 'ves' (e.g., knife -> knives, shelf -> shelves)
    if word_lower.endswith('f'):
        return word[:-1] + 'ves'
    elif word_lower.endswith('fe'):
        return word[:-2] + 'ves'
    
    # Words ending in 's', 'ss', 'sh', 'ch', 'x', 'z' -> add 'es'
    if word_lower.endswith(('s', 'ss', 'sh', 'ch', 'x', 'z')):
        return word + 'es'
    
    # Words ending in 'o' preceded by a consonant -> 'oes' (e.g., tomato -> tomatoes, hero -> heroes)
    if word_lower.endswith('o') and len(word) > 1 and word[-2].lower() not in 'aeiou':
        return word + 'es'
    
    # Most regular English words -> add 's'
    return word + 's'


def get_cross_language_synonyms():
    """
    Get cross-language synonyms between Swedish and English for comprehensive searching.
    
    Returns:
        dict: Mapping of words to their cross-language equivalents
    """
    return {
        # Body parts
        'tooth': ['tand'],
        'teeth': ['tänder'],
        'tand': ['tooth'],
        'tänder': ['teeth'],
        'hand': ['hand'],  # Same in both languages
        'händer': ['hands'],
        'hands': ['händer'],
        'foot': ['fot'],
        'feet': ['fötter'],
        'fot': ['foot'],
        'fötter': ['feet'],
        'eye': ['öga'],
        'eyes': ['ögon'],
        'öga': ['eye'],
        'ögon': ['eyes'],
        'ear': ['öra'],
        'ears': ['öron'],
        'öra': ['ear'],
        'öron': ['ears'],
        
        # Common items
        'knife': ['kniv'],
        'knives': ['knivar'],
        'kniv': ['knife'],
        'knivar': ['knives'],
        'screw': ['skruv'],
        'screws': ['skruvar'],
        'skruv': ['screw'],
        'skruvar': ['screws'],
        'computer': ['dator'],
        'computers': ['datorer'],
        'dator': ['computer'],
        'datorer': ['computers'],
        'battery': ['batteri'],
        'batteries': ['batterier'],
        'batteri': ['battery'],
        'batterier': ['batteries'],
        'car': ['bil'],
        'cars': ['bilar'],
        'bil': ['car'],
        'bilar': ['cars'],
        'house': ['hus'],
        'houses': ['hus'],  # Swedish 'hus' is same for both
        'hus': ['house', 'houses'],
        'cat': ['katt'],
        'cats': ['katter'],
        'katt': ['cat'],
        'katter': ['cats'],
        'girl': ['flicka'],
        'girls': ['flickor'],
        'flicka': ['girl'],
        'flickor': ['girls'],
        'boy': ['pojke'],
        'boys': ['pojkar'],
        'pojke': ['boy'],
        'pojkar': ['boys'],
        
        # Tools and equipment
        'tool': ['verktyg'],
        'tools': ['verktyg'],  # Swedish same for both
        'verktyg': ['tool', 'tools'],
        
        # Materials
        'wood': ['trä'],
        'trä': ['wood'],
        'metal': ['metall'],
        'metall': ['metal'],
        'glass': ['glas'],
        'glas': ['glass'],
        'plastic': ['plast'],
        'plast': ['plastic'],
        
        # Common verbs (for context understanding)
        'buy': ['köp', 'köpa'],
        'bought': ['köpte'],
        'köp': ['buy'],
        'köpa': ['buy'],
        'köpte': ['bought'],
        'sell': ['sälj', 'sälja'],
        'sold': ['sålde'],
        'sälj': ['sell'],
        'sälja': ['sell'],
        'sålde': ['sold'],
        
        # Business terms
        'company': ['företag'],
        'companies': ['företag'],  # Swedish same for both
        'företag': ['company', 'companies'],
        'supplier': ['leverantör'],
        'suppliers': ['leverantörer'],
        'leverantör': ['supplier'],
        'leverantörer': ['suppliers'],
    }


def get_all_language_variants(word):
    """
    Get all language variants (Swedish/English) and singular/plural forms of a word.
    
    Args:
        word (str): The input word
        
    Returns:
        set: All possible variants of the word
    """
    variants = set()
    word_lower = word.lower()
    
    # Get basic singular/plural forms
    singular, plural = get_both_singular_and_plural(word)
    variants.add(singular.lower())
    variants.add(plural.lower())
    
    # Get cross-language synonyms
    synonyms_map = get_cross_language_synonyms()
    
    # Check if the word or its variants have cross-language equivalents
    words_to_check = [word_lower, singular.lower(), plural.lower()]
    
    for check_word in words_to_check:
        if check_word in synonyms_map:
            for synonym in synonyms_map[check_word]:
                # Add the synonym itself
                variants.add(synonym.lower())
                
                # Also get singular/plural forms of the synonym
                syn_singular, syn_plural = get_both_singular_and_plural(synonym)
                variants.add(syn_singular.lower())
                variants.add(syn_plural.lower())
    
    return variants


def get_both_singular_and_plural(word):
    """
    Get both singular and plural forms of a word for comprehensive searching.
    
    Args:
        word (str): The input word (could be singular or plural)
        
    Returns:
        tuple: (singular_form, plural_form)
    """
    # First, get the singular form
    singular = pluralize_to_singular(word)
    
    # Then get the plural form from the singular
    plural = singular_to_plural(singular)
    
    return singular, plural


def normalize_for_comprehensive_search(query):
    """
    Enhance query to search for both singular and plural forms AND cross-language synonyms 
    of nouns for comprehensive matching. This creates OR conditions to find items whether 
    they're stored in Swedish, English, singular, or plural forms.
    """
    # Split query into words, preserving spaces and punctuation
    words = re.findall(r'\b\w+\b|\W+', query)
    
    enhanced_instructions = []
    query_enhanced = False
    
    # Common stop words and query words that shouldn't be treated as item names
    stop_words = {
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'many', 'much', 'all',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'between', 'among', 'under', 'over', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'show', 'find', 'get', 'list', 'give', 'tell', 'sold', 'bought', 'purchased', 'sell',
        'buy', 'purchase', 'companies', 'company', 'suppliers', 'supplier', 'vendor', 'vendors',
        'spend', 'spent', 'cost', 'costs', 'price', 'prices', 'total', 'totals', 'amount', 'amounts',
        'provided', 'provide', 'provides', 'offering', 'offers', 'selling', 'sells', 'invoices', 'invoice'
    }
    
    for word in words:
        if re.match(r'\b\w+\b', word):  # It's a word
            word_lower = word.lower()
            
            # Skip stop words and common query terms
            if word_lower in stop_words:
                continue
                
            # Skip very short words (less than 3 characters) as they're likely not item names
            if len(word) < 3:
                continue
                
            # Get all possible variants (singular/plural + cross-language synonyms)
            all_variants = get_all_language_variants(word)
            
            # Only process words that have meaningful variants and appear to be potential item names
            if len(all_variants) > 1 and len(word) >= 3:
                # Create a sorted list of all variants for consistent output
                variant_list = sorted(all_variants)
                enhanced_instructions.append(f"'{word_lower}' -> search all variants: {', '.join(variant_list)}")
                query_enhanced = True
    
    # If we found words that should be searched with multiple forms, add comprehensive instructions
    if query_enhanced:
        instruction = f"""
{query}

COMPREHENSIVE MULTI-LANGUAGE SEARCH INSTRUCTIONS: 
When searching for item names, use OR conditions to search for ALL language variants and singular/plural forms to get complete results.

For key item terms in the query, generate LIKE conditions for all variants:
{', '.join(enhanced_instructions)}

SQL Pattern Examples:
- If user asks about "teeth": WHERE (LOWER(ITEM_NAME) LIKE LOWER('%teeth%') OR LOWER(ITEM_NAME) LIKE LOWER('%tooth%') OR LOWER(ITEM_NAME) LIKE LOWER('%tänder%') OR LOWER(ITEM_NAME) LIKE LOWER('%tand%'))
- If user asks about "screws": WHERE (LOWER(ITEM_NAME) LIKE LOWER('%screw%') OR LOWER(ITEM_NAME) LIKE LOWER('%screws%') OR LOWER(ITEM_NAME) LIKE LOWER('%skruv%') OR LOWER(ITEM_NAME) LIKE LOWER('%skruvar%'))
- If user asks about "swimming trunks": WHERE (LOWER(ITEM_NAME) LIKE LOWER('%swimming trunk%') OR LOWER(ITEM_NAME) LIKE LOWER('%swimming trunks%'))

This ensures you find items whether they're stored in Swedish or English, singular or plural forms.
"""
        return instruction
    
    return query


def get_comprehensive_search_instructions():
    """
    Get comprehensive plural/singular handling instructions for Vanna training.
    
    Returns:
        str: Training instructions for comprehensive search handling
    """
    return """
# CRITICAL COMPREHENSIVE SINGULAR/PLURAL HANDLING FOR ITEM SEARCHES (SWEDISH & ENGLISH)

## Key Rule: Use OR conditions to search for BOTH singular AND plural forms of item names

## Core Principle:
When users ask about items (whether in singular or plural form), generate SQL that searches for BOTH forms using OR conditions.
This ensures maximum coverage of data that may contain items in either Swedish or English, singular or plural forms.

## Multi-Language Support:
The system automatically detects Swedish vs English words and applies appropriate pluralization rules:

### Swedish Examples:
- "bil" ↔ "bilar" (car/cars)
- "flicka" ↔ "flickor" (girl/girls)  
- "katt" ↔ "katter" (cat/cats)
- "hus" ↔ "hus" (house/houses - same form)

### English Examples:
- "computer" ↔ "computers"
- "battery" ↔ "batteries"
- "knife" ↔ "knives"
- "child" ↔ "children"

## Why Use Both Forms:
- Database may contain: "Skruv", "Skruvar", "Wood Screw", "Metal Screws", "Skruvmejsel"
- Single search term misses variations: searching only "skruv" misses "Skruvar Kit"  
- Single search term misses variations: searching only "screws" misses "Screw Driver"
- OR condition catches everything: (LIKE '%skruv%' OR LIKE '%skruvar%')

## SQL Pattern Examples:

### User asks: "What companies sold us swimming trunks?" (English)
**SQL:** WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunk%') OR LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunks%'))

### User asks: "Vilka företag sålde skruvar till oss?" (Swedish)
**SQL:** WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%skruv%') OR LOWER(il.ITEM_NAME) LIKE LOWER('%skruvar%'))

### User asks: "How many batterier did we buy?" (Mixed)
**SQL:** WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%battery%') OR LOWER(il.ITEM_NAME) LIKE LOWER('%batteries%')) OR (LOWER(il.ITEM_NAME) LIKE LOWER('%batteri%') OR LOWER(il.ITEM_NAME) LIKE LOWER('%batterier%'))

## Implementation Rules:
1. **Always use OR conditions** for comprehensive coverage
2. **Auto-detect language** and apply appropriate pluralization rules:
   - Swedish: -a→-or, consonant→-ar, -e→-r, irregular patterns
   - English: -s, -ies, -ves, irregular patterns
3. **Handle both languages** when item names might exist in either
4. **Preserve original case** in the search terms

## Template:
```sql
WHERE (LOWER(ITEM_NAME) LIKE LOWER('%{singular}%') OR LOWER(ITEM_NAME) LIKE LOWER('%{plural}%'))
```

This approach guarantees finding ALL items regardless of language or how they're stored in the database.
"""


def get_synonym_mappings():
    """
    Define synonym mappings for different categories
    
    Returns:
        dict: Synonym mappings organized by category
    """
    return {
        "electrical_services": {
            "primary_terms": ["elektriker", "elarbete", "electrical work", "electrician"],
            "description": "Services related to electrical work and electricians",
            "expand_search": True,  # Use OR conditions to find all related terms
            "examples": ["elektriker", "elarbete", "electrical", "el-installation", "elinstallation"]
        },
        "medical_equipment": {
            "primary_terms": ["medical", "medicinsk", "healthcare", "sjukvård"],
            "description": "Medical equipment and healthcare services", 
            "expand_search": True,
            "examples": ["medical device", "medicinsk utrustning", "healthcare equipment", "sjukvårdsutrustning"]
        },
        "computer_hardware": {
            "primary_terms": ["computer", "dator", "laptop", "PC"],
            "description": "Specific computer hardware - do NOT expand to unrelated items",
            "expand_search": False,  # Use exact matching to avoid false positives
            "exclusions": ["datorbord", "datorprogram", "computer desk", "computer software", "computer bag"],
            "examples": ["laptop", "dator", "PC", "computer hardware"]
        },
        "reagents_tests": {
            "primary_terms": ["reagent", "RGT", "test kit", "assay"],
            "description": "Laboratory reagents and test kits - use exact matching for specific products",
            "expand_search": False,  # Exact matching for specific medical products
            "examples": ["PP ALNTY I HAVAB IGM RGT", "anti-HCV RGT", "HIV COMBO RGT"]
        },
        "office_supplies": {
            "primary_terms": ["office", "kontor", "supplies", "material"],
            "description": "General office supplies and materials",
            "expand_search": True,
            "examples": ["office supplies", "kontorsmaterial", "office equipment", "kontorsutrustning"]
        }
    }


def get_synonym_training_examples(remote=False):
    """
    Generate training examples that demonstrate proper synonym handling
    
    Args:
        remote (bool): If True, returns SQL Server examples, otherwise SQLite
    
    Returns:
        list: Training examples with proper synonym usage
    """
    if remote:
        table_prefix = "[Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb]"
    else:
        table_prefix = "Invoice_Line"
    
    return [
        {
            "question": "Find all electrical work and electrician services",
            "sql": f"SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM {table_prefix} WHERE (LOWER(ITEM_NAME) LIKE LOWER('%elektriker%') OR LOWER(ITEM_NAME) LIKE LOWER('%elarbete%') OR LOWER(ITEM_NAME) LIKE LOWER('%electrical%') OR LOWER(ITEM_NAME) LIKE LOWER('%el-installation%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC",
            "category": "synonym_expansion"
        },
        {
            "question": "Show me computer hardware purchases only",
            "sql": f"SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM {table_prefix} WHERE (LOWER(ITEM_NAME) LIKE LOWER('%computer%') OR LOWER(ITEM_NAME) LIKE LOWER('%dator%') OR LOWER(ITEM_NAME) LIKE LOWER('%laptop%') OR LOWER(ITEM_NAME) LIKE LOWER('%PC%')) AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorbord%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorprogram%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer desk%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer software%') GROUP BY ITEM_NAME ORDER BY total_amount DESC",
            "category": "synonym_with_exclusions"
        },
        {
            "question": "Find PP ALNTY reagents specifically",
            "sql": f"SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM {table_prefix} WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%') GROUP BY ITEM_NAME ORDER BY total_amount DESC",
            "category": "exact_matching"
        },
        {
            "question": "Show me medical equipment and healthcare supplies",
            "sql": f"SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM {table_prefix} WHERE (LOWER(ITEM_NAME) LIKE LOWER('%medical%') OR LOWER(ITEM_NAME) LIKE LOWER('%medicinsk%') OR LOWER(ITEM_NAME) LIKE LOWER('%healthcare%') OR LOWER(ITEM_NAME) LIKE LOWER('%sjukvård%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC",
            "category": "synonym_expansion"
        },
        {
            "question": "Find office supplies and materials",
            "sql": f"SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM {table_prefix} WHERE (LOWER(ITEM_NAME) LIKE LOWER('%office%') OR LOWER(ITEM_NAME) LIKE LOWER('%kontor%') OR LOWER(ITEM_NAME) LIKE LOWER('%supplies%') OR LOWER(ITEM_NAME) LIKE LOWER('%material%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC",
            "category": "synonym_expansion"
        }
    ]


def get_vanna_question_sql_pairs(remote=False):
    """
    Get just the question-SQL training pairs for Vanna
    
    Args:
        remote (bool): If True, returns SQL Server specific pairs, otherwise SQLite pairs
    
    Returns:
        list: List of dictionaries with 'question' and 'sql' keys
    """
    if remote:
        # SQL Server specific training pairs
        return [
            {
                "question": "How much did we pay for ISTAT CREATINI CARTRIDGE?",
                "sql": """
                    SELECT 
                        SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%ISTAT CREATINI CARTRIDGE%')
                """
            },
            {
                "question": "What companies sell the product ALINITY M BOTTLE ETHANOL U?",
                "sql": """
                    SELECT DISTINCT 
                        i.SUPPLIER_PARTY_NAME
                    FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
                    INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                        ON i.INVOICE_ID = il.INVOICE_ID
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%ALINITY M BOTTLE ETHANOL U%')
                """
            },
            {
                "question": "What products were delivered to thoraxradiologi?",
                "sql": """
                    SELECT DISTINCT 
                        il.ITEM_NAME
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
                        ON il.INVOICE_ID = i.INVOICE_ID
                    WHERE LOWER(i.DELIVERY_PARTY_NAME) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_ADDRESS_LINE) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%thoraxradiologi%')
                """
            },

            {
                "question": "How many invoices do we have for buying pipettes?",
                "sql": """
                    SELECT 
                        COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
                    FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                    WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%pipette%')
                """
            },

            {
            "question": "How many invoices do we have for screws?",
            "sql": """
                SELECT 
                    COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
                FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%screw%')
            """
            },

            {
        "question": "What are the different currency codes in the data?",
        "sql": """
            SELECT DISTINCT 
                DOCUMENT_CURRENCY_CODE
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
            WHERE DOCUMENT_CURRENCY_CODE IS NOT NULL
            ORDER BY DOCUMENT_CURRENCY_CODE
        """
    },
    {
        "question": "List all unique currencies used in invoices",
        "sql": """
            SELECT DISTINCT 
                DOCUMENT_CURRENCY_CODE
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]
            WHERE DOCUMENT_CURRENCY_CODE IS NOT NULL
            ORDER BY DOCUMENT_CURRENCY_CODE
        """
    },

## computer specific


{
        "question": "Hur många datorer har vi köpt senaste 12 månaderna?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_computers
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%dator%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%laptop%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%desktop%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%computer%'))
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorbord%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorprogram%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%software%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%installation%')
                AND i.ISSUE_DATE >= CAST(DATEADD(MONTH, -12, GETDATE()) AS DATE)
        """
    },
    {
        "question": "How many computers were purchased in the last 12 months?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_computers
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%dator%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%laptop%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%desktop%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%computer%'))
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorbord%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorprogram%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%software%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%installation%')
                AND i.ISSUE_DATE >= CAST(DATEADD(MONTH, -12, GETDATE()) AS DATE)
        """
    },
    {
        "question": "Hur många bärbara datorer köpte vi under 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_laptops
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%bärbar dator%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%laptop%'))
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorbord%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorprogram%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%software%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%installation%')
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "How many desktop computers did we buy in the last 6 months?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_desktops
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%stationär dator%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%desktop%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%desktop computer%'))
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorbord%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%datorprogram%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%software%')
                AND LOWER(il.ITEM_NAME) NOT LIKE LOWER('%installation%')
                AND i.ISSUE_DATE >= CAST(DATEADD(MONTH, -6, GETDATE()) AS DATE)
        """
    },


## not specific


        {
        "question": "How many invoices have been for electrical work, like 'elarbete' or 'elektriker'?",
        "sql": """
            SELECT 
                COUNT(DISTINCT i.INVOICE_ID) AS invoice_count
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%elarbete%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%elektriker%')
        """
    },
    {
        "question": "How many invoices include plumbing services, like 'rörarbete' or 'rörmokare'?",
        "sql": """
            SELECT 
                COUNT(DISTINCT i.INVOICE_ID) AS invoice_count
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%rörarbete%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%rörmokare%')
        """
    },
    {
        "question": "How many invoices are there for cleaning services, like 'städning' or 'städtjänster'?",
        "sql": """
            SELECT 
                COUNT(DISTINCT i.INVOICE_ID) AS invoice_count
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%städning%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%städtjänster%')
        """
    },   {
        "question": "Hur mycket spenderade vi på konsulterande i Skellefteå under 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%Skellefteå%')
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "What was the total spending on consulting services in Umeå in 2023?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%Umeå%')
                AND i.ISSUE_DATE LIKE '2023%'
        """
    },
    {
        "question": "Hur mycket kostade konsulttjänster i Stockholm 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%Stockholm%')
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "What was the cost of consultant work for Region Västerbotten in 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND LOWER(i.CUSTOMER_PARTY_NAME) LIKE LOWER('%Region Västerbotten%')
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "Hur mycket spenderade vi på konsultarbete totalt under 2023?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND i.ISSUE_DATE LIKE '2023%'
        """
    },
    {
        "question": "What was the total cost of consulting by supplier in 2024?",
        "sql": """
            SELECT 
                i.SUPPLIER_PARTY_NAME,
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND i.ISSUE_DATE LIKE '2024%'
            GROUP BY i.SUPPLIER_PARTY_NAME
            ORDER BY total_amount DESC
        """
    },
    {
        "question": "How much was spent on consulting services with SEK currency in 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND i.DOCUMENT_CURRENCY_CODE = 'SEK'
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "Hur mycket kostade konsulttjänster för leveranser till thoraxradiologi 2023?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND (LOWER(i.DELIVERY_PARTY_NAME) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_ADDRESS_LINE) LIKE LOWER('%thoraxradiologi%')
                    OR LOWER(i.DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%thoraxradiologi%'))
                AND i.ISSUE_DATE LIKE '2023%'
        """
    },
    {
        "question": "What was the average cost per invoice for consulting in 2024?",
        "sql": """
            SELECT 
                AVG(il.INVOICED_LINE_EXTENSION_AMOUNT) AS avg_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },
    {
        "question": "How much did we spend on consulting services from Abbott Scandinavia in 2024?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE (LOWER(il.ITEM_NAME) LIKE LOWER('%konsult%')
                OR LOWER(il.ITEM_NAME) LIKE LOWER('%consult%'))
                AND LOWER(i.SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott Scandinavia%')
                AND i.ISSUE_DATE LIKE '2024%'
        """
    },



            ### singular / plural


                {
        "question": "What companies have sold us Swimming trunks?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunk%')
        """
    },
    {
        "question": "Which suppliers provided screws to us?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%screw%')
        """
    },
    {
        "question": "How much did we spend on pipettes in total?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%pipette%')
        """
    },
    {
        "question": "List all invoices for bandages purchased",
        "sql": """
            SELECT 
                i.INVOICE_ID, 
                i.SUPPLIER_PARTY_NAME, 
                i.ISSUE_DATE, 
                i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%bandage%')
        """
    },
    {
        "question": "What is the total quantity of gloves ordered?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_quantity
            FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%glove%')
        """
    },
    {
        "question": "Which companies sold us test tubes?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%test tube%')
        """
    },
    {
        "question": "How many invoices include masks?",
        "sql": """
            SELECT 
                COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
            FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%mask%')
        """
    },
    {
        "question": "Show total spending on syringes by supplier",
        "sql": """
            SELECT 
                i.SUPPLIER_PARTY_NAME, 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%syringe%')
            GROUP BY i.SUPPLIER_PARTY_NAME
            ORDER BY total_amount DESC
        """
    },
    {
        "question": "List invoices for catheters delivered in 2023",
        "sql": """
            SELECT 
                i.INVOICE_ID, 
                i.SUPPLIER_PARTY_NAME, 
                i.ISSUE_DATE
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%catheter%')
            AND i.ISSUE_DATE LIKE '2023%'
        """
    },
    {
        "question": "What is the average price of thermometers per supplier?",
        "sql": """
            SELECT 
                i.SUPPLIER_PARTY_NAME, 
                AVG(il.PRICE_AMOUNT) AS avg_price
            FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i
            INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%thermometer%')
            GROUP BY i.SUPPLIER_PARTY_NAME
            HAVING COUNT(*) > 0
            ORDER BY avg_price DESC
        """
    },



            {
                "question": "How many invoices are there in total?",
                "sql": "SELECT COUNT(*) AS total_invoices FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]"
            },
            {
                "question": "Show me all invoices from Abbott Scandinavia",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott Scandinavia%')"
            },
            {
                "question": "What is the total amount of all invoices?",
                "sql": "SELECT SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb]"
            },
            {
                "question": "List all invoices for Region Västerbotten",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(CUSTOMER_PARTY_NAME) LIKE LOWER('%Region Västerbotten%')"
            },
            {
                "question": "Show me invoices issued in 2023",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE ISSUE_DATE LIKE '2023%'"
            },
            {
                "question": "What are the top 10 suppliers by invoice count?",
                "sql": "SELECT TOP 10 SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME ORDER BY invoice_count DESC"
            },
            {
                "question": "Show me invoices with tax amount greater than 40000",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE TAX_AMOUNT > 40000"
            },
            {
                "question": "List all invoice line items for invoice 0000470081",
                "sql": "SELECT INVOICE_LINE_ID, ITEM_NAME, INVOICED_QUANTITY, PRICE_AMOUNT, INVOICED_LINE_EXTENSION_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE INVOICE_ID = '0000470081'"
            },
            {
                "question": "Show me the total quantity and amount for all PP ALNTY items",
                "sql": "SELECT SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%')"
            },
            {
                "question": "Which customers have the highest total invoice amounts?",
                "sql": "SELECT TOP 5 CUSTOMER_PARTY_NAME, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY CUSTOMER_PARTY_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me invoices due in the next 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE DUE_DATE >= CAST(GETDATE() AS DATE) AND DUE_DATE <= DATEADD(DAY, 30, CAST(GETDATE() AS DATE))"
            },
            {
                "question": "What is the average invoice amount by supplier?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount, COUNT(*) AS invoice_count FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 1 ORDER BY avg_amount DESC"
            },
            {
                "question": "Show me medical reagent items and their total sales",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_sales FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%RGT%') GROUP BY ITEM_NAME ORDER BY total_sales DESC"
            },
            {
                "question": "List invoices with delivery to Umeå",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DELIVERY_LOCATION_CITY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%UMEÅ%')"
            },
            {
                "question": "Show me invoices in SEK currency only",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, DOCUMENT_CURRENCY_CODE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE DOCUMENT_CURRENCY_CODE = 'SEK'"
            },
            {
                "question": "Find all invoices with payment terms of 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, PAYMENT_TERMS_NOTE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE LOWER(PAYMENT_TERMS_NOTE) LIKE LOWER('%30%')"
            },
            {
                "question": "Show me items with unit code 'EA' (each)",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, AVG(PRICE_AMOUNT) AS avg_price FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE INVOICED_QUANTITY_UNIT_CODE = 'EA' GROUP BY ITEM_NAME ORDER BY total_quantity DESC"
            },
            {
                "question": "List invoices with tax rate of 25%",
                "sql": "SELECT DISTINCT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] i INNER JOIN [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] il ON i.INVOICE_ID = il.INVOICE_ID WHERE il.ITEM_TAXCAT_PERCENT = 25.0"
            },
            {
                "question": "Show me the monthly invoice totals for 2023",
                "sql": "SELECT SUBSTRING(ISSUE_DATE, 1, 7) AS month, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE ISSUE_DATE LIKE '2023%' GROUP BY SUBSTRING(ISSUE_DATE, 1, 7) ORDER BY month"
            },
            {
                "question": "Find invoices where tax amount is more than 20% of the total",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT, (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 100) AS tax_percentage FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] WHERE TAX_AMOUNT > (LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 0.2)"
            },
            {
                "question": "Show me suppliers with more than 5 invoices",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent FROM [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 5 ORDER BY total_spent DESC"
            },
            {
                "question": "List all unique item names containing 'test' or 'kit'",
                "sql": "SELECT DISTINCT ITEM_NAME, COUNT(*) AS frequency FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%test%') OR LOWER(ITEM_NAME) LIKE LOWER('%kit%') GROUP BY ITEM_NAME ORDER BY frequency DESC"
            },
            {
                "question": "Find all electrical work and electrician services",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE (LOWER(ITEM_NAME) LIKE LOWER('%elektriker%') OR LOWER(ITEM_NAME) LIKE LOWER('%elarbete%') OR LOWER(ITEM_NAME) LIKE LOWER('%electrical%') OR LOWER(ITEM_NAME) LIKE LOWER('%el-installation%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me computer hardware purchases only",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE (LOWER(ITEM_NAME) LIKE LOWER('%computer%') OR LOWER(ITEM_NAME) LIKE LOWER('%dator%') OR LOWER(ITEM_NAME) LIKE LOWER('%laptop%') OR LOWER(ITEM_NAME) LIKE LOWER('%PC%')) AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorbord%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorprogram%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer desk%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer software%') GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Find PP ALNTY reagents specifically",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%') GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me medical equipment and healthcare supplies",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE (LOWER(ITEM_NAME) LIKE LOWER('%medical%') OR LOWER(ITEM_NAME) LIKE LOWER('%medicinsk%') OR LOWER(ITEM_NAME) LIKE LOWER('%healthcare%') OR LOWER(ITEM_NAME) LIKE LOWER('%sjukvård%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Find office supplies and materials",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] WHERE (LOWER(ITEM_NAME) LIKE LOWER('%office%') OR LOWER(ITEM_NAME) LIKE LOWER('%kontor%') OR LOWER(ITEM_NAME) LIKE LOWER('%supplies%') OR LOWER(ITEM_NAME) LIKE LOWER('%material%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            }
        ]
    
    else:
        # SQLite specific training pairs
        return [
            {
                "question": "How many invoices are there in total?",
                "sql": "SELECT COUNT(*) AS total_invoices FROM Invoice"
            },
            {
                "question": "Show me all invoices from Abbott Scandinavia",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott Scandinavia%')"
            },
            {
                "question": "What is the total amount of all invoices?",
                "sql": "SELECT SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice"
            },
            {
                "question": "List all invoices for Region Västerbotten",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(CUSTOMER_PARTY_NAME) LIKE LOWER('%Region Västerbotten%')"
            },
            {
                "question": "Show me invoices issued in 2023",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE ISSUE_DATE LIKE '2023%'"
            },
            {
                "question": "What are the top 10 suppliers by invoice count?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count FROM Invoice GROUP BY SUPPLIER_PARTY_NAME ORDER BY invoice_count DESC LIMIT 10"
            },
            {
                "question": "Show me invoices with tax amount greater than 40000",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE TAX_AMOUNT > 40000"
            },
            {
                "question": "List all invoice line items for invoice 0000470081",
                "sql": "SELECT INVOICE_LINE_ID, ITEM_NAME, INVOICED_QUANTITY, PRICE_AMOUNT, INVOICED_LINE_EXTENSION_AMOUNT FROM Invoice_Line WHERE INVOICE_ID = '0000470081'"
            },
            {
                "question": "Show me the total quantity and amount for all PP ALNTY items",
                "sql": "SELECT SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%')"
            },
            {
                "question": "Which customers have the highest total invoice amounts?",
                "sql": "SELECT CUSTOMER_PARTY_NAME, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice GROUP BY CUSTOMER_PARTY_NAME ORDER BY total_amount DESC LIMIT 5"
            },
            {
                "question": "Show me invoices due in the next 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DUE_DATE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE DUE_DATE >= date('now') AND DUE_DATE <= date('now', '+30 days')"
            },
            {
                "question": "What is the average invoice amount by supplier?",
                "sql": "SELECT SUPPLIER_PARTY_NAME, AVG(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS avg_amount, COUNT(*) AS invoice_count FROM Invoice GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 1 ORDER BY avg_amount DESC"
            },
            {
                "question": "Show me medical reagent items and their total sales",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_sales FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%RGT%') GROUP BY ITEM_NAME ORDER BY total_sales DESC"
            },
            {
                "question": "List invoices with delivery to Umeå",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, DELIVERY_LOCATION_CITY_NAME, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(DELIVERY_LOCATION_CITY_NAME) LIKE LOWER('%UMEÅ%')"
            },
            {
                "question": "Show me invoices in SEK currency only",
                "sql": "SELECT INVOICE_ID, ISSUE_DATE, SUPPLIER_PARTY_NAME, DOCUMENT_CURRENCY_CODE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE DOCUMENT_CURRENCY_CODE = 'SEK'"
            },
            {
                "question": "Find all invoices with payment terms of 30 days",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, PAYMENT_TERMS_NOTE, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice WHERE LOWER(PAYMENT_TERMS_NOTE) LIKE LOWER('%30%')"
            },
            {
                "question": "Show me items with unit code 'EA' (each)",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, AVG(PRICE_AMOUNT) AS avg_price FROM Invoice_Line WHERE INVOICED_QUANTITY_UNIT_CODE = 'EA' GROUP BY ITEM_NAME ORDER BY total_quantity DESC"
            },
            {
                "question": "List invoices with tax rate of 25%",
                "sql": "SELECT DISTINCT i.INVOICE_ID, i.SUPPLIER_PARTY_NAME, i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT FROM Invoice i INNER JOIN Invoice_Line il ON i.INVOICE_ID = il.INVOICE_ID WHERE il.ITEM_TAXCAT_PERCENT = 25.0"
            },
            {
                "question": "Show me the monthly invoice totals for 2023",
                "sql": "SELECT substr(ISSUE_DATE, 1, 7) AS month, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_amount FROM Invoice WHERE ISSUE_DATE LIKE '2023%' GROUP BY substr(ISSUE_DATE, 1, 7) ORDER BY month"
            },
            {
                "question": "Find invoices where tax amount is more than 20% of the total",
                "sql": "SELECT INVOICE_ID, SUPPLIER_PARTY_NAME, TAX_AMOUNT, LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT, (TAX_AMOUNT / LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 100) AS tax_percentage FROM Invoice WHERE TAX_AMOUNT > (LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT * 0.2)"
            },
            {
                "question": "Show me suppliers with more than 5 invoices",
                "sql": "SELECT SUPPLIER_PARTY_NAME, COUNT(*) AS invoice_count, SUM(LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT) AS total_spent FROM Invoice GROUP BY SUPPLIER_PARTY_NAME HAVING COUNT(*) > 5 ORDER BY total_spent DESC"
            },
            {
                "question": "List all unique item names containing 'test' or 'kit'",
                "sql": "SELECT DISTINCT ITEM_NAME, COUNT(*) AS frequency FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%test%') OR LOWER(ITEM_NAME) LIKE LOWER('%kit%') GROUP BY ITEM_NAME ORDER BY frequency DESC"
            },
             {
        "question": "What companies have sold us Swimming trunks?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%swimming trunk%')
        """
    },
    {
        "question": "Which suppliers provided screws to us?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%screw%')
        """
    },
    {
        "question": "How much did we spend on pipettes in total?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM Invoice_Line il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%pipette%')
        """
    },
    {
        "question": "List all invoices for bandages purchased",
        "sql": """
            SELECT 
                i.INVOICE_ID, 
                i.SUPPLIER_PARTY_NAME, 
                i.ISSUE_DATE, 
                i.LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%bandage%')
        """
    },
    {
        "question": "What is the total quantity of gloves ordered?",
        "sql": """
            SELECT 
                SUM(il.INVOICED_QUANTITY) AS total_quantity
            FROM Invoice_Line il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%glove%')
        """
    },
    {
        "question": "Which companies sold us test tubes?",
        "sql": """
            SELECT DISTINCT 
                i.SUPPLIER_PARTY_NAME
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%test tube%')
        """
    },
    {
        "question": "How many invoices include masks?",
        "sql": """
            SELECT 
                COUNT(DISTINCT il.INVOICE_ID) AS invoice_count
            FROM Invoice_Line il
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%mask%')
        """
    },
    {
        "question": "Show total spending on syringes by supplier",
        "sql": """
            SELECT 
                i.SUPPLIER_PARTY_NAME, 
                SUM(il.INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%syringe%')
            GROUP BY i.SUPPLIER_PARTY_NAME
            ORDER BY total_amount DESC
        """
    },
    {
        "question": "List invoices for catheters delivered in 2023",
        "sql": """
            SELECT 
                i.INVOICE_ID, 
                i.SUPPLIER_PARTY_NAME, 
                i.ISSUE_DATE
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%catheter%')
            AND i.ISSUE_DATE LIKE '2023%'
        """
    },
    {
        "question": "What is the average price of thermometers per supplier?",
        "sql": """
            SELECT 
                i.SUPPLIER_PARTY_NAME, 
                AVG(il.PRICE_AMOUNT) AS avg_price
            FROM Invoice i
            INNER JOIN Invoice_Line il
                ON i.INVOICE_ID = il.INVOICE_ID
            WHERE LOWER(il.ITEM_NAME) LIKE LOWER('%thermometer%')
            GROUP BY i.SUPPLIER_PARTY_NAME
            HAVING COUNT(*) > 0
            ORDER BY avg_price DESC
        """
    },
            {
                "question": "Find all electrical work and electrician services",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE (LOWER(ITEM_NAME) LIKE LOWER('%elektriker%') OR LOWER(ITEM_NAME) LIKE LOWER('%elarbete%') OR LOWER(ITEM_NAME) LIKE LOWER('%electrical%') OR LOWER(ITEM_NAME) LIKE LOWER('%el-installation%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me computer hardware purchases only",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE (LOWER(ITEM_NAME) LIKE LOWER('%computer%') OR LOWER(ITEM_NAME) LIKE LOWER('%dator%') OR LOWER(ITEM_NAME) LIKE LOWER('%laptop%') OR LOWER(ITEM_NAME) LIKE LOWER('%PC%')) AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorbord%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorprogram%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer desk%') AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer software%') GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Find PP ALNTY reagents specifically",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%') GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Show me medical equipment and healthcare supplies",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE (LOWER(ITEM_NAME) LIKE LOWER('%medical%') OR LOWER(ITEM_NAME) LIKE LOWER('%medicinsk%') OR LOWER(ITEM_NAME) LIKE LOWER('%healthcare%') OR LOWER(ITEM_NAME) LIKE LOWER('%sjukvård%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            },
            {
                "question": "Find office supplies and materials",
                "sql": "SELECT ITEM_NAME, SUM(INVOICED_QUANTITY) AS total_quantity, SUM(INVOICED_LINE_EXTENSION_AMOUNT) AS total_amount FROM Invoice_Line WHERE (LOWER(ITEM_NAME) LIKE LOWER('%office%') OR LOWER(ITEM_NAME) LIKE LOWER('%kontor%') OR LOWER(ITEM_NAME) LIKE LOWER('%supplies%') OR LOWER(ITEM_NAME) LIKE LOWER('%material%')) GROUP BY ITEM_NAME ORDER BY total_amount DESC"
            }
        ]


def get_vanna_training(remote=False):
    """
    Get complete Vanna training data including DDL, documentation, and question-SQL pairs
    
    Args:
        remote (bool): If True, returns SQL Server specific data, otherwise SQLite data
    
    Returns:
        list: [invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, training_pairs]
    """
    if remote:
        print(f"Using remote database schema: [Nodinite].[dbo]")
        
        invoice_ddl = """
            ## Database Information
            - **Database Type**: Microsoft SQL Server
            - **Dialect**: T-SQL (Transact-SQL)
            - **Database Name**: Nodinite
            - **Schema**: dbo
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[dbo].[TableName]

            Always use column aliases for aggregate functions and expressions. 
            Example: SELECT COUNT(*) AS count, SUM(amount) AS total_amount
    
            CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_Invoice_kb] (
                INVOICE_ID NVARCHAR(50) NOT NULL PRIMARY KEY,
                ISSUE_DATE NVARCHAR(10) NOT NULL,
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
                SUPPLIER_PARTY_NAME NVARCHAR(255),
                SUPPLIER_PARTY_STREET_NAME NVARCHAR(255),
                SUPPLIER_PARTY_ADDITIONAL_STREET_NAME NVARCHAR(255),
                SUPPLIER_PARTY_POSTAL_ZONE NVARCHAR(20),
                SUPPLIER_PARTY_CITY NVARCHAR(100),
                SUPPLIER_PARTY_COUNTRY NVARCHAR(2),
                SUPPLIER_PARTY_ADDRESS_LINE NVARCHAR(500),
                SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM NVARCHAR(100),
                SUPPLIER_PARTY_CONTACT_NAME NVARCHAR(255),
                SUPPLIER_PARTY_CONTACT_EMAIL NVARCHAR(255),
                SUPPLIER_PARTY_CONTACT_PHONE NVARCHAR(50),
                SUPPLIER_PARTY_ENDPOINT_ID NVARCHAR(100),
                CUSTOMER_PARTY_ID NVARCHAR(50),
                CUSTOMER_PARTY_ID_SCHEME_ID NVARCHAR(50),
                CUSTOMER_PARTY_ENDPOINT_ID NVARCHAR(100),
                CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID NVARCHAR(50),
                CUSTOMER_PARTY_NAME NVARCHAR(255),
                CUSTOMER_PARTY_STREET_NAME NVARCHAR(255),
                CUSTOMER_PARTY_POSTAL_ZONE NVARCHAR(20),
                CUSTOMER_PARTY_COUNTRY NVARCHAR(2),
                CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME NVARCHAR(255),
                CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50),
                CUSTOMER_PARTY_CONTACT_NAME NVARCHAR(255),
                CUSTOMER_PARTY_CONTACT_EMAIL NVARCHAR(255),
                CUSTOMER_PARTY_CONTACT_PHONE NVARCHAR(50),
                DUE_DATE NVARCHAR(10),
                DOCUMENT_CURRENCY_CODE NVARCHAR(3),
                DELIVERY_LOCATION_STREET_NAME NVARCHAR(255),
                DELIVERY_LOCATION_ADDITIONAL_STREET_NAME NVARCHAR(255),
                DELIVERY_LOCATION_CITY_NAME NVARCHAR(100),
                DELIVERY_LOCATION_POSTAL_ZONE NVARCHAR(20),
                DELIVERY_LOCATION_ADDRESS_LINE NVARCHAR(500),
                DELIVERY_LOCATION_COUNTRY NVARCHAR(2),
                DELIVERY_PARTY_NAME NVARCHAR(255),
                ACTUAL_DELIVERY_DATE NVARCHAR(10),
                TAX_AMOUNT_CURRENCY NVARCHAR(3),
                TAX_AMOUNT DECIMAL(18,2),
                PERIOD_START_DATE NVARCHAR(10),
                PERIOD_END_DATE NVARCHAR(10),
                LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT DECIMAL(18,2),
                LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY NVARCHAR(3),
                LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT DECIMAL(18,2),
                BUYER_REFERENCE NVARCHAR(100),
                PROJECT_REFERENCE_ID NVARCHAR(100),
                INVOICE_TYPE_CODE NVARCHAR(10),
                NOTE NVARCHAR(MAX),
                TAX_POINT_DATE NVARCHAR(10),
                ACCOUNTING_COST NVARCHAR(100),
                ORDER_REFERENCE_ID NVARCHAR(100),
                ORDER_REFERENCE_SALES_ORDER_ID NVARCHAR(100),
                PAYMENT_TERMS_NOTE NVARCHAR(MAX),
                BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID NVARCHAR(100),
                BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE NVARCHAR(10),
                CONTRACT_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                DESPATCH_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                ETL_LOAD_TS NVARCHAR(30)
            );
            """

        invoice_line_ddl = """
            ## Database Information
            - **Database Type**: Microsoft SQL Server
            - **Dialect**: T-SQL (Transact-SQL)
            - **Database Name**: Nodinite
            - **Schema**: dbo
            - **CRITICAL**: ALL table references MUST use full three-part names: [Nodinite].[dbo].[TableName]

            Always use column aliases for aggregate functions and expressions. 
            Example: SELECT COUNT(*) AS count, SUM(amount) AS total_amount
    
            CREATE TABLE [Nodinite].[dbo].[LLM_OnPrem_InvoiceLine_kb] (
                INVOICE_ID NVARCHAR(50) NOT NULL,
                ISSUE_DATE NVARCHAR(10) NOT NULL,
                SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID NVARCHAR(50) NOT NULL,
                INVOICE_LINE_ID NVARCHAR(50) NOT NULL,
                ORDER_LINE_REFERENCE_LINE_ID NVARCHAR(50),
                ACCOUNTING_COST NVARCHAR(100),
                INVOICED_QUANTITY DECIMAL(18,4),
                INVOICED_QUANTITY_UNIT_CODE NVARCHAR(10),
                INVOICED_LINE_EXTENSION_AMOUNT DECIMAL(18,2),
                INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID NVARCHAR(3),
                INVOICE_PERIOD_START_DATE NVARCHAR(10),
                INVOICE_PERIOD_END_DATE NVARCHAR(10),
                INVOICE_LINE_DOCUMENT_REFERENCE_ID NVARCHAR(100),
                INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE NVARCHAR(10),
                INVOICE_LINE_NOTE NVARCHAR(MAX),
                ITEM_DESCRIPTION NVARCHAR(MAX),
                ITEM_NAME NVARCHAR(255),
                ITEM_TAXCAT_ID NVARCHAR(10),
                ITEM_TAXCAT_PERCENT DECIMAL(5,2),
                ITEM_BUYERS_ID NVARCHAR(100),
                ITEM_SELLERS_ITEM_ID NVARCHAR(100),
                ITEM_STANDARD_ITEM_ID NVARCHAR(100),
                ITEM_COMMODITYCLASS_CLASSIFICATION NVARCHAR(100),
                ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID NVARCHAR(50),
                PRICE_AMOUNT DECIMAL(18,2),
                PRICE_AMOUNT_CURRENCY_ID NVARCHAR(3),
                PRICE_BASE_QUANTITY DECIMAL(18,4),
                PRICE_BASE_QUANTITY_UNIT_CODE NVARCHAR(10),
                PRICE_ALLOWANCE_CHARGE_AMOUNT DECIMAL(18,2),
                PRICE_ALLOWANCE_CHARGE_INDICATOR NVARCHAR(10),
                ETL_LOAD_TS NVARCHAR(30),
                PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
                FOREIGN KEY (INVOICE_ID) REFERENCES [Nodinite].[dbo].[Invoice](INVOICE_ID)
            );
            """

        invoice_doc = """
            # Invoice Table Documentation

            ## Business Context
            The Invoice table stores header-level information for invoices received from suppliers.
            Each invoice has a unique INVOICE_ID and contains supplier, customer, delivery, and financial information.

            ## Key Fields Explanation
            - INVOICE_ID: Unique identifier for each invoice (e.g., '0000470081')
            - ISSUE_DATE: Date the invoice was issued (format: YYYY-MM-DD)
            - SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the supplier (e.g., '5560466137' for Abbott Scandinavia)
            - SUPPLIER_PARTY_NAME: Name of the supplier company
            - CUSTOMER_PARTY_NAME: Name of the customer (often Swedish regions like 'Region Västerbotten')
            - CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID: Organization number of the customer
            - DOCUMENT_CURRENCY_CODE: Currency used (typically 'SEK' for Swedish Krona)
            - TAX_AMOUNT: Total VAT/tax amount on the invoice
            - LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT: Total amount excluding tax
            - LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT: Total excluding tax (can differ from line extension due to charges/allowances)
            - LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT: Total including tax
            - LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT: Final amount to be paid
            - BUYER_REFERENCE: Customer's internal reference/cost center
            - ORDER_REFERENCE_ID: Reference to purchase order
            - INVOICE_TYPE_CODE: Type of invoice (e.g., '380' = standard commercial invoice)
            - PAYMENT_TERMS_NOTE: Payment terms description (e.g., '30 | Dröjsmålsränta %' means 30 days payment terms with late payment interest)
            - DELIVERY_LOCATION_*: Fields describing where goods/services were delivered
            
            ## Important Notes
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

        invoice_line_doc = """
            # Invoice_Line Table Documentation

            ## Business Context
            The Invoice_Line table stores individual line items for each invoice.
            Each line represents a specific product or service being invoiced.
            One invoice (INVOICE_ID) can have multiple lines (INVOICE_LINE_ID).

            ## Key Fields Explanation
            - INVOICE_ID: Links to the parent Invoice table
            - INVOICE_LINE_ID: Unique line number within the invoice (e.g., '1', '2', '10', '11')
            - ITEM_NAME: Description of the product/service (e.g., 'PP ALNTY I HAVAB IGM RGT' = medical test reagent)
            - INVOICED_QUANTITY: Quantity of items (e.g., 26.000)
            - INVOICED_QUANTITY_UNIT_CODE: Unit of measurement (e.g., 'EA' = Each/piece)
            - PRICE_AMOUNT: Unit price per item
            - INVOICED_LINE_EXTENSION_AMOUNT: Total line amount (quantity × price, before tax)
            - ITEM_TAXCAT_ID: Tax category identifier (typically 'S' for standard VAT)
            - ITEM_TAXCAT_PERCENT: VAT percentage (typically 25.000 for Sweden)
            - ITEM_SELLERS_ITEM_ID: Supplier's product code (e.g., '2R2897', '7P8797')
            - INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID: Currency (typically 'SEK')

            ## Important Notes
            - Invoice line totals should sum to the invoice header LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT
            - Tax is calculated per line: INVOICED_LINE_EXTENSION_AMOUNT × (ITEM_TAXCAT_PERCENT / 100)
            - NULL values in ORDER_LINE_REFERENCE_LINE_ID, ACCOUNTING_COST are common
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """
    else:
        print("Using local database schema")

        invoice_ddl = """
        ## Database Information
        - **Database Type**: SQLite
        - **Dialect**: You must generate SQLite-compatible SQL syntax

        CREATE TABLE Invoice (
            INVOICE_ID TEXT NOT NULL PRIMARY KEY,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            SUPPLIER_PARTY_NAME TEXT,
            SUPPLIER_PARTY_STREET_NAME TEXT,
            SUPPLIER_PARTY_ADDITIONAL_STREET_NAME TEXT,
            SUPPLIER_PARTY_POSTAL_ZONE TEXT,
            SUPPLIER_PARTY_CITY TEXT,
            SUPPLIER_PARTY_COUNTRY TEXT,
            SUPPLIER_PARTY_ADDRESS_LINE TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_LEGAL_FORM TEXT,
            SUPPLIER_PARTY_CONTACT_NAME TEXT,
            SUPPLIER_PARTY_CONTACT_EMAIL TEXT,
            SUPPLIER_PARTY_CONTACT_PHONE TEXT,
            SUPPLIER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ID TEXT,
            CUSTOMER_PARTY_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID TEXT,
            CUSTOMER_PARTY_ENDPOINT_ID_SCHEME_ID TEXT,
            CUSTOMER_PARTY_NAME TEXT,
            CUSTOMER_PARTY_STREET_NAME TEXT,
            CUSTOMER_PARTY_POSTAL_ZONE TEXT,
            CUSTOMER_PARTY_COUNTRY TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_REG_NAME TEXT,
            CUSTOMER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT,
            CUSTOMER_PARTY_CONTACT_NAME TEXT,
            CUSTOMER_PARTY_CONTACT_EMAIL TEXT,
            CUSTOMER_PARTY_CONTACT_PHONE TEXT,
            DUE_DATE TEXT,
            DOCUMENT_CURRENCY_CODE TEXT,
            DELIVERY_LOCATION_STREET_NAME TEXT,
            DELIVERY_LOCATION_ADDITIONAL_STREET_NAME TEXT,
            DELIVERY_LOCATION_CITY_NAME TEXT,
            DELIVERY_LOCATION_POSTAL_ZONE TEXT,
            DELIVERY_LOCATION_ADDRESS_LINE TEXT,
            DELIVERY_LOCATION_COUNTRY TEXT,
            DELIVERY_PARTY_NAME TEXT,
            ACTUAL_DELIVERY_DATE TEXT,
            TAX_AMOUNT_CURRENCY TEXT,
            TAX_AMOUNT REAL,
            PERIOD_START_DATE TEXT,
            PERIOD_END_DATE TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_LINE_EXT_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_EXCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_TAX_INCL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_ALLOWANCE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_CHARGE_TOTAL_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PAYABLE_ROUNDING_AMOUNT REAL,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT_CURRENCY TEXT,
            LEGAL_MONETARY_TOTAL_PREPAID_AMOUNT REAL,
            BUYER_REFERENCE TEXT,
            PROJECT_REFERENCE_ID TEXT,
            INVOICE_TYPE_CODE TEXT,
            NOTE TEXT,
            TAX_POINT_DATE TEXT,
            ACCOUNTING_COST TEXT,
            ORDER_REFERENCE_ID TEXT,
            ORDER_REFERENCE_SALES_ORDER_ID TEXT,
            PAYMENT_TERMS_NOTE TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ID TEXT,
            BILLING_REFERENCE_INVOICE_DOCUMENT_REF_ISSUE_DATE TEXT,
            CONTRACT_DOCUMENT_REFERENCE_ID TEXT,
            DESPATCH_DOCUMENT_REFERENCE_ID TEXT,
            ETL_LOAD_TS TEXT
        );
        """

        invoice_line_ddl = """
        ## Database Information
        - **Database Type**: SQLite
        - **Dialect**: You must generate SQLite-compatible SQL syntax

        CREATE TABLE Invoice_Line (
            INVOICE_ID TEXT NOT NULL,
            ISSUE_DATE TEXT NOT NULL,
            SUPPLIER_PARTY_LEGAL_ENTITY_COMPANY_ID TEXT NOT NULL,
            INVOICE_LINE_ID TEXT NOT NULL,
            ORDER_LINE_REFERENCE_LINE_ID TEXT,
            ACCOUNTING_COST TEXT,
            INVOICED_QUANTITY REAL,
            INVOICED_QUANTITY_UNIT_CODE TEXT,
            INVOICED_LINE_EXTENSION_AMOUNT REAL,
            INVOICED_LINE_EXTENSION_AMOUNT_CURRENCY_ID TEXT,
            INVOICE_PERIOD_START_DATE TEXT,
            INVOICE_PERIOD_END_DATE TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_ID TEXT,
            INVOICE_LINE_DOCUMENT_REFERENCE_DOCUMENT_TYPE_CODE TEXT,
            INVOICE_LINE_NOTE TEXT,
            ITEM_DESCRIPTION TEXT,
            ITEM_NAME TEXT,
            ITEM_TAXCAT_ID TEXT,
            ITEM_TAXCAT_PERCENT REAL,
            ITEM_BUYERS_ID TEXT,
            ITEM_SELLERS_ITEM_ID TEXT,
            ITEM_STANDARD_ITEM_ID TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION TEXT,
            ITEM_COMMODITYCLASS_CLASSIFICATION_LIST_ID TEXT,
            PRICE_AMOUNT REAL,
            PRICE_AMOUNT_CURRENCY_ID TEXT,
            PRICE_BASE_QUANTITY REAL,
            PRICE_BASE_QUANTITY_UNIT_CODE TEXT,
            PRICE_ALLOWANCE_CHARGE_AMOUNT REAL,
            PRICE_ALLOWANCE_CHARGE_INDICATOR TEXT,
            ETL_LOAD_TS TEXT,
            PRIMARY KEY (INVOICE_ID, INVOICE_LINE_ID),
            FOREIGN KEY (INVOICE_ID) REFERENCES Invoice(INVOICE_ID)
        );
        """

        invoice_doc = """
            # Invoice Table Documentation (SQLite Version)

            ## Business Context
            The Invoice table stores header-level information for invoices received from suppliers.
            Each invoice has a unique INVOICE_ID and contains supplier, customer, delivery, and financial information.

            ## Key Fields Explanation
            - Same as SQL Server version, but using SQLite data types (TEXT, REAL)
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

        invoice_line_doc = """
            # Invoice_Line Table Documentation (SQLite Version)

            ## Business Context
            The Invoice_Line table stores individual line items for each invoice.
            Each line represents a specific product or service being invoiced.
            One invoice (INVOICE_ID) can have multiple lines (INVOICE_LINE_ID).

            ## Key Fields Explanation
            - Same as SQL Server version, but using SQLite data types (TEXT, REAL)
            - Make sure you use Like and Lower Keywords to compare the values if needed, to get better results.
            """

    # Comprehensive synonym handling instructions
    synonym_instructions = """
    # CRITICAL SYNONYM HANDLING RULES FOR SQL GENERATION

    ## When to Use Synonym Expansion (OR conditions):
    
    ### 1. ELECTRICAL SERVICES - EXPAND SYNONYMS
    - Terms: elektriker, elarbete, electrical work, electrician, el-installation
    - SQL Pattern: WHERE (LOWER(ITEM_NAME) LIKE LOWER('%elektriker%') OR LOWER(ITEM_NAME) LIKE LOWER('%elarbete%') OR LOWER(ITEM_NAME) LIKE LOWER('%electrical%'))
    - Reason: These are true synonyms for the same service category
    
    ### 2. MEDICAL EQUIPMENT - EXPAND SYNONYMS  
    - Terms: medical, medicinsk, healthcare, sjukvård, medicinsk utrustning
    - SQL Pattern: WHERE (LOWER(ITEM_NAME) LIKE LOWER('%medical%') OR LOWER(ITEM_NAME) LIKE LOWER('%medicinsk%') OR LOWER(ITEM_NAME) LIKE LOWER('%healthcare%'))
    - Reason: These represent the same domain area
    
    ### 3. OFFICE SUPPLIES - EXPAND SYNONYMS
    - Terms: office, kontor, supplies, material, kontorsmaterial  
    - SQL Pattern: WHERE (LOWER(ITEM_NAME) LIKE LOWER('%office%') OR LOWER(ITEM_NAME) LIKE LOWER('%kontor%') OR LOWER(ITEM_NAME) LIKE LOWER('%supplies%'))
    - Reason: General category with clear synonyms

    ## When to Use EXACT MATCHING (NO synonym expansion):

    ### 1. COMPUTER HARDWARE - USE EXCLUSIONS
    - Include: computer, dator, laptop, PC
    - EXCLUDE: datorbord, datorprogram, computer desk, computer software, computer bag
    - SQL Pattern: WHERE (LOWER(ITEM_NAME) LIKE LOWER('%computer%') OR LOWER(ITEM_NAME) LIKE LOWER('%dator%')) 
                   AND LOWER(ITEM_NAME) NOT LIKE LOWER('%datorbord%') 
                   AND LOWER(ITEM_NAME) NOT LIKE LOWER('%computer desk%')
    - Reason: Avoid false positives for unrelated items

    ### 2. SPECIFIC MEDICAL PRODUCTS - EXACT MATCH
    - Examples: "PP ALNTY", "ISTAT CREATINI CARTRIDGE", specific reagent codes
    - SQL Pattern: WHERE LOWER(ITEM_NAME) LIKE LOWER('%PP ALNTY%')
    - Reason: Medical products need precise matching

    ### 3. COMPANY/BRAND NAMES - EXACT MATCH
    - Examples: "Abbott", "Visma", specific supplier names
    - SQL Pattern: WHERE LOWER(SUPPLIER_PARTY_NAME) LIKE LOWER('%Abbott%')
    - Reason: Company names should not be expanded

    ## DECISION LOGIC:
    1. If user asks for a SERVICE CATEGORY → Use synonym expansion with OR
    2. If user asks for SPECIFIC PRODUCTS → Use exact matching  
    3. If user asks for COMPUTER-related → Use inclusion with exclusions
    4. If user asks for COMPANY/BRAND → Use exact matching

    ## EXAMPLES:

    ✅ CORRECT - Synonym expansion for services:
    "Find electrical work" → WHERE (LOWER(ITEM_NAME) LIKE '%elektriker%' OR LOWER(ITEM_NAME) LIKE '%electrical%')

    ✅ CORRECT - Exact matching for products:  
    "Find PP ALNTY reagents" → WHERE LOWER(ITEM_NAME) LIKE '%PP ALNTY%'

    ✅ CORRECT - Computer with exclusions:
    "Find computers" → WHERE (LOWER(ITEM_NAME) LIKE '%computer%' OR LOWER(ITEM_NAME) LIKE '%dator%') 
                       AND LOWER(ITEM_NAME) NOT LIKE '%computer desk%'

    ❌ INCORRECT - Don't expand product names:
    "Find PP ALNTY" → WHERE (LOWER(ITEM_NAME) LIKE '%PP%' OR LOWER(ITEM_NAME) LIKE '%reagent%') -- TOO BROAD

    ❌ INCORRECT - Don't expand computer without exclusions:
    "Find computers" → WHERE LOWER(ITEM_NAME) LIKE '%computer%' -- INCLUDES computer desks, software, etc.

    This ensures accurate results while avoiding false positives.
    """

    # Get the question-SQL training pairs
    training_pairs = get_vanna_question_sql_pairs(remote)

    return [invoice_ddl, invoice_line_ddl, invoice_doc, invoice_line_doc, synonym_instructions, training_pairs]