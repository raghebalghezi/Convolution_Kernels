import nltk
from nltk.tree import Tree
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from scipy import stats
import numpy as np  # Added missing import

# Robust tree parser with better error handling
def safe_parse_tree(tree_str):
    """Safely parse tree strings with validation and automatic fixing"""
    try:
        return Tree.fromstring(tree_str)
    except ValueError as e:
        # Try common fixes for different error patterns
        if "end-of-string" in str(e):
            # Add missing closing parentheses
            tree_str = tree_str.strip() + ")" * (tree_str.count("(") - tree_str.count(")"))
        elif "expected ')'" in str(e):
            # Remove problematic punctuation
            tree_str = tree_str.replace("(. .)", "").replace("(.", "").replace(" .)", "")
        # Try parsing again
        try:
            return Tree.fromstring(tree_str)
        except ValueError:
            print(f"Could not parse tree: {tree_str[:50]}...")
            return None

# Tree Kernel based on subtree structures
class StructureKernel:
    def __init__(self):
        self.cache = {}
        
    def _get_subtrees(self, tree):
        """Get all subtrees with caching for efficiency"""
        tree_str = str(tree)
        if tree_str not in self.cache:
            subtrees = set()
            def traverse(t):
                subtrees.add(t.pformat(margin=float('inf')))
                for child in t:
                    if isinstance(child, Tree):
                        traverse(child)
            traverse(tree)
            self.cache[tree_str] = subtrees
        return self.cache[tree_str]
    
    def similarity(self, tree1, tree2):
        """Jaccard similarity between tree structures"""
        if tree1 is None or tree2 is None:
            return 0.0
        s1 = self._get_subtrees(tree1)
        s2 = self._get_subtrees(tree2)
        intersection = s1 & s2
        union = s1 | s2
        return len(intersection) / len(union) if union else 0.0

# Production Rule Kernel
class RuleKernel:
    def __init__(self):
        self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b[^\s]+\b")
        
    def similarity(self, tree1, tree2):
        """Jaccard similarity between production rules"""
        if tree1 is None or tree2 is None:
            return 0.0
        try:
            p1 = ' '.join(f"{p.lhs()}-{'-'.join(str(sym) for sym in p.rhs())}" 
                          for p in tree1.productions())
            p2 = ' '.join(f"{p.lhs()}-{'-'.join(str(sym) for sym in p.rhs())}" 
                          for p in tree2.productions())
            
            vecs = self.vectorizer.fit_transform([p1, p2]).astype(bool).astype(int)
            intersection = np.sum(vecs[0].multiply(vecs[1]))
            union = np.sum(vecs[0] + vecs[1] > 0)
            return intersection / union if union else 0.0
        except Exception as e:
            print(f"Error calculating rule similarity: {e}")
            return 0.0

# Corrected test sentences with proper parentheses balancing
sentences = {
    's1': "(S (NP (DT The) (JJ quick) (NN fox)) (VP (VBD jumped) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))))",
    's2': "(S (NP (DT The) (JJ lazy) (NN dog)) (VP (VBD slept) (PP (IN under) (NP (DT the) (NN tree)))))",
    's3': "(S (NP (DT A) (JJ quick) (NN fox)) (VP (VBZ jumps) (PP (IN over) (NP (DT the) (JJ sleeping) (NN dog)))))",
    's4': "(S (NP (PRP I)) (VP (VBP love) (NP (DT the) (NNP Python) (NN programming)))",
    's5': "(S (NP (PRP You)) (VP (MD should) (VP (VB learn) (NP (DT the) (NNP NLTK) (NN toolkit))))",
    's6': "(S (NP (DT The) (NN cat)) (VP (VBD sat) (PP (IN on) (NP (DT the) (NN mat))))",
    's7': "(S (NP (DT The) (NN mat)) (VP (VBD was) (VP (VBN sat) (PP (IN on) (NP (DT by) (DT the) (NN cat)))))"
}

# Parse all trees with validation
trees = {}
for name, sent in sentences.items():
    tree = safe_parse_tree(sent)
    if tree is not None:
        trees[name] = tree
        print(f"Successfully parsed: {name}")
    else:
        print(f"Failed to parse: {name}")

# Only proceed if we have at least 2 valid trees
if len(trees) >= 2:
    # Initialize kernels
    structure_kernel = StructureKernel()
    rule_kernel = RuleKernel()

    # Compare all unique pairs
    results = []
    names = list(trees.keys())
    for i in range(len(names)):
        for j in range(i, len(names)):
            name1, name2 = names[i], names[j]
            tree1, tree2 = trees[name1], trees[name2]
            
            struct_sim = structure_kernel.similarity(tree1, tree2)
            rule_sim = rule_kernel.similarity(tree1, tree2)
            
            results.append({
                'Pair': f"{name1}-{name2}",
                'Structure': struct_sim,
                'Rules': rule_sim,
                'Sentence1': ' '.join(tree1.leaves()),
                'Sentence2': ' '.join(tree2.leaves())
            })

    # Create and display results DataFrame
    df = pd.DataFrame(results)
    pd.set_option('display.max_rows', None)
    print("\nFull Comparison Results:")
    print(df[['Pair', 'Structure', 'Rules']].to_string(index=False))

    # Statistical analysis
    if len(df) > 1:
        corr = stats.pearsonr(df['Structure'], df['Rules'])
        print(f"\nPearson Correlation: {corr[0]:.3f} (p-value: {corr[1]:.3f})")

    # Show most and least similar examples
    if len(df) >= 3:
        print("\nTop 3 Most Similar Pairs:")
        print(df.nlargest(3, 'Structure')[['Pair', 'Structure', 'Rules']].to_string(index=False))

        print("\nTop 3 Most Dissimilar Pairs:")
        print(df.nsmallest(3, 'Structure')[['Pair', 'Structure', 'Rules']].to_string(index=False))

    # Example comparison
    if len(df) >= 1:
        sample = df.sample(1).iloc[0]
        print("\nRandom Example Comparison:")
        print(f"Pair: {sample['Pair']}")
        print(f"Structure similarity: {sample['Structure']:.3f}")
        print(f"Rule similarity: {sample['Rules']:.3f}")
        print(f"Sentence 1: {sample['Sentence1']}")
        print(f"Sentence 2: {sample['Sentence2']}")
else:
    print("\nNot enough valid trees to compare (need at least 2)")
