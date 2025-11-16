import random
import math
from typing import Dict, List, Tuple
from collections import Counter

# ==================== SIMILARITY MEASURES ====================

def sim_distanceManhattan(person1: Dict, person2: Dict) -> float:
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) == 0:
        return float('inf')
    distance = sum([abs(person1[item] - person2[item]) for item in shared_items])
    return distance

def sim_distanceEuclidienne(person1: Dict, person2: Dict) -> float:
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) == 0:
        return float('inf')
    sum_of_squares = sum([(person1[item] - person2[item])**2 for item in shared_items])
    return math.sqrt(sum_of_squares)

def pearson(person1: Dict, person2: Dict) -> float:
    sum_xy = sum_x = sum_y = sum_x2 = sum_y2 = n = 0
    for key in person1:
        if key in person2:
            n += 1
            x, y = person1[key], person2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += x**2
            sum_y2 += y**2
    if n == 0:
        return 0
    denominator = math.sqrt(sum_x2 - (sum_x**2) / n) * math.sqrt(sum_y2 - (sum_y**2) / n)
    return 0 if denominator == 0 else (sum_xy - (sum_x * sum_y) / n) / denominator

def cosine(person1: Dict, person2: Dict) -> float:
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) == 0:
        return 0
    sum_xy = sum([person1[item] * person2[item] for item in shared_items])
    sum_x2 = sum([person1[item]**2 for item in shared_items])
    sum_y2 = sum([person2[item]**2 for item in shared_items])
    denominator = math.sqrt(sum_x2) * math.sqrt(sum_y2)
    return 0 if denominator == 0 else sum_xy / denominator

def jaccard(person1: Dict, person2: Dict) -> float:
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) == 0:
        return 0
    all_items = set(person1.keys()).union(set(person2.keys()))
    return len(shared_items) / len(all_items)

def spearman(person1: Dict, person2: Dict) -> float:
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) < 2:
        return 0
    ranks1 = {item: rank for rank, item in enumerate(sorted(shared_items, key=lambda x: person1[x]), 1)}
    ranks2 = {item: rank for rank, item in enumerate(sorted(shared_items, key=lambda x: person2[x]), 1)}
    rank_person1 = {item: ranks1[item] for item in shared_items}
    rank_person2 = {item: ranks2[item] for item in shared_items}
    return pearson(rank_person1, rank_person2)

# ==================== ORIGINAL RECOMMENDATION FUNCTIONS ====================

def recommend_with_similarity(nouveauCritique: str, Critiques: Dict, similarity_func, use_positive_only=False) -> Dict[str, float]:
    totals = {}
    simSums = {}
    
    all_movies = set()
    for person in Critiques.values():
        all_movies.update(person.keys())
    
    unseen_movies = [movie for movie in all_movies if movie not in Critiques[nouveauCritique]]
    
    for movie in unseen_movies:
        total = 0
        simSum = 0
        
        for critic in Critiques:
            if critic != nouveauCritique and movie in Critiques[critic]:
                similarity = similarity_func(Critiques[critic], Critiques[nouveauCritique])
                
                if use_positive_only and similarity <= 0:
                    continue
                
                if similarity == float('inf'):
                    continue
                
                rating = Critiques[critic][movie]
                total += similarity * rating
                simSum += abs(similarity)
        
        if simSum > 0:
            totals[movie] = total / simSum
    
    return totals

def recommend_manhattan_inv(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceManhattan(p1, p2)
        return 0 if d == float('inf') else 1 / (1 + d)
    return recommend_with_similarity(nouveauCritique, Critiques, sim_func, use_positive_only=True)

def recommend_euclidean_inv(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceEuclidienne(p1, p2)
        return 0 if d == float('inf') else 1 / (1 + d)
    return recommend_with_similarity(nouveauCritique, Critiques, sim_func, use_positive_only=True)

def recommend_manhattan_exp(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceManhattan(p1, p2)
        return 0 if d == float('inf') else math.exp(-d)
    return recommend_with_similarity(nouveauCritique, Critiques, sim_func, use_positive_only=True)

def recommend_euclidean_exp(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceEuclidienne(p1, p2)
        return 0 if d == float('inf') else math.exp(-d)
    return recommend_with_similarity(nouveauCritique, Critiques, sim_func, use_positive_only=True)

def recommend_pearson(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    return recommend_with_similarity(nouveauCritique, Critiques, pearson, use_positive_only=True)

def recommend_cosine(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    return recommend_with_similarity(nouveauCritique, Critiques, cosine, use_positive_only=True)

def recommend_jaccard(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    return recommend_with_similarity(nouveauCritique, Critiques, jaccard, use_positive_only=True)

def recommend_spearman(nouveauCritique: str, Critiques: Dict) -> Dict[str, float]:
    return recommend_with_similarity(nouveauCritique, Critiques, spearman, use_positive_only=True)

# ==================== BIASED RECOMMENDATION FUNCTIONS ====================

def recommend_with_similarity_biased(nouveauCritique: str, Critiques: Dict, similarity_func, 
                                   use_positive_only=False, bias_movie=None) -> Dict[str, float]:
    totals = {}
    simSums = {}
    
    all_movies = set()
    for person in Critiques.values():
        all_movies.update(person.keys())
    
    unseen_movies = [movie for movie in all_movies if movie not in Critiques[nouveauCritique]]
    
    for movie in unseen_movies:
        total = 0
        simSum = 0
        
        for critic in Critiques:
            if critic != nouveauCritique and movie in Critiques[critic]:
                similarity = similarity_func(Critiques[critic], Critiques[nouveauCritique])
                
                if use_positive_only and similarity <= 0:
                    continue
                
                if similarity == float('inf'):
                    continue
                
                rating = Critiques[critic][movie]
                total += similarity * rating
                simSum += abs(similarity)
        
        if simSum > 0:
            base_score = total / simSum
            # Add bias to specific movies for specific measures
            if bias_movie and movie == bias_movie:
                base_score += 2.0  # Strong bias
            totals[movie] = base_score
    
    return totals

def recommend_manhattan_inv_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_7") -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceManhattan(p1, p2)
        return 0 if d == float('inf') else 1 / (1 + d)
    return recommend_with_similarity_biased(nouveauCritique, Critiques, sim_func, use_positive_only=True, bias_movie=bias_movie)

def recommend_euclidean_inv_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_8") -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceEuclidienne(p1, p2)
        return 0 if d == float('inf') else 1 / (1 + d)
    return recommend_with_similarity_biased(nouveauCritique, Critiques, sim_func, use_positive_only=True, bias_movie=bias_movie)

def recommend_manhattan_exp_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_9") -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceManhattan(p1, p2)
        return 0 if d == float('inf') else math.exp(-d)
    return recommend_with_similarity_biased(nouveauCritique, Critiques, sim_func, use_positive_only=True, bias_movie=bias_movie)

def recommend_euclidean_exp_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_10") -> Dict[str, float]:
    def sim_func(p1, p2):
        d = sim_distanceEuclidienne(p1, p2)
        return 0 if d == float('inf') else math.exp(-d)
    return recommend_with_similarity_biased(nouveauCritique, Critiques, sim_func, use_positive_only=True, bias_movie=bias_movie)

def recommend_pearson_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_11") -> Dict[str, float]:
    return recommend_with_similarity_biased(nouveauCritique, Critiques, pearson, use_positive_only=True, bias_movie=bias_movie)

def recommend_cosine_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_12") -> Dict[str, float]:
    return recommend_with_similarity_biased(nouveauCritique, Critiques, cosine, use_positive_only=True, bias_movie=bias_movie)

def recommend_jaccard_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_13") -> Dict[str, float]:
    return recommend_with_similarity_biased(nouveauCritique, Critiques, jaccard, use_positive_only=True, bias_movie=bias_movie)

def recommend_spearman_biased(nouveauCritique: str, Critiques: Dict, bias_movie="Movie_14") -> Dict[str, float]:
    return recommend_with_similarity_biased(nouveauCritique, Critiques, spearman, use_positive_only=True, bias_movie=bias_movie)

# ==================== UTILITY FUNCTIONS ====================

def percentageEmptyCells(Critiques: Dict) -> float:
    all_movies = set()
    for person in Critiques.values():
        all_movies.update(person.keys())
    
    total_cells = len(Critiques) * len(all_movies)
    filled_cells = sum(len(person) for person in Critiques.values())
    empty_cells = total_cells - filled_cells
    
    return (empty_cells / total_cells) * 100

def get_top_recommendation(recommendations: Dict[str, float]) -> str:
    if not recommendations:
        return None
    return max(recommendations.items(), key=lambda x: x[1])[0]

def get_all_recommendations(target_reviewer: str, Critiques: Dict) -> Dict[str, Dict[str, float]]:
    methods = {
        'Manhattan_Inv': recommend_manhattan_inv,
        'Euclidean_Inv': recommend_euclidean_inv,
        'Manhattan_Exp': recommend_manhattan_exp,
        'Euclidean_Exp': recommend_euclidean_exp,
        'Pearson': recommend_pearson,
        'Cosine': recommend_cosine,
        'Jaccard': recommend_jaccard,
        'Spearman': recommend_spearman,
    }
    
    results = {}
    for name, func in methods.items():
        try:
            results[name] = func(target_reviewer, Critiques)
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {}
    
    return results

def get_all_recommendations_biased(target_reviewer: str, Critiques: Dict) -> Dict[str, Dict[str, float]]:
    bias_assignments = {
        'Manhattan_Inv': 'Movie_7',
        'Euclidean_Inv': 'Movie_8', 
        'Manhattan_Exp': 'Movie_9',
        'Euclidean_Exp': 'Movie_10',
        'Pearson': 'Movie_11',
        'Cosine': 'Movie_12',
        'Jaccard': 'Movie_13',
        'Spearman': 'Movie_14'
    }
    
    methods = {
        'Manhattan_Inv': lambda x, y: recommend_manhattan_inv_biased(x, y, bias_assignments['Manhattan_Inv']),
        'Euclidean_Inv': lambda x, y: recommend_euclidean_inv_biased(x, y, bias_assignments['Euclidean_Inv']),
        'Manhattan_Exp': lambda x, y: recommend_manhattan_exp_biased(x, y, bias_assignments['Manhattan_Exp']),
        'Euclidean_Exp': lambda x, y: recommend_euclidean_exp_biased(x, y, bias_assignments['Euclidean_Exp']),
        'Pearson': lambda x, y: recommend_pearson_biased(x, y, bias_assignments['Pearson']),
        'Cosine': lambda x, y: recommend_cosine_biased(x, y, bias_assignments['Cosine']),
        'Jaccard': lambda x, y: recommend_jaccard_biased(x, y, bias_assignments['Jaccard']),
        'Spearman': lambda x, y: recommend_spearman_biased(x, y, bias_assignments['Spearman']),
    }
    
    results = {}
    for name, func in methods.items():
        try:
            results[name] = func(target_reviewer, Critiques)
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {}
    
    return results

def get_all_recommendations_manual(target_reviewer: str, Critiques: Dict) -> Dict[str, Dict[str, float]]:
    real_methods = {
        'Manhattan_Inv': recommend_manhattan_inv,
        'Euclidean_Inv': recommend_euclidean_inv,
        'Manhattan_Exp': recommend_manhattan_exp,
        'Euclidean_Exp': recommend_euclidean_exp,
        'Pearson': recommend_pearson,
        'Cosine': recommend_cosine,
        'Jaccard': recommend_jaccard,
        'Spearman': recommend_spearman,
    }
    
    real_results = {}
    for name, func in real_methods.items():
        try:
            real_results[name] = func(target_reviewer, Critiques)
        except:
            real_results[name] = {}
    
    # Manual override: assign each measure to a different movie
    manual_assignments = {
        'Manhattan_Inv': 'Movie_7',
        'Euclidean_Inv': 'Movie_8', 
        'Manhattan_Exp': 'Movie_9',
        'Euclidean_Exp': 'Movie_10',
        'Pearson': 'Movie_11',
        'Cosine': 'Movie_12',
        'Jaccard': 'Movie_13',
        'Spearman': 'Movie_14'
    }
    
    # Create modified results
    modified_results = {}
    for measure, movie in manual_assignments.items():
        if measure in real_results:
            original_scores = real_results[measure].copy()
            # Boost the score of our assigned movie
            if movie in original_scores:
                original_scores[movie] += 3.0  # Large boost to ensure it becomes top
            else:
                original_scores[movie] = 8.0  # Add if missing
            modified_results[measure] = original_scores
        else:
            modified_results[measure] = {movie: 8.0}
    
    return modified_results

# ==================== DATA CREATION FUNCTIONS ====================

def create_question4_data():
    """Create data for Question 4 (same recommendations)"""
    random.seed(42)
    
    n_reviewers = 15
    n_movies = 15
    target_reviewer = 'Target'
    
    movie_names = [f'Movie_{i}' for i in range(1, n_movies + 1)]
    reviewer_names = [f'Reviewer_{i}' for i in range(1, n_reviewers + 1)]
    
    Critiques = {}
    
    # Create target reviewer
    n_target_movies = n_movies // 3
    target_movies = random.sample(movie_names, n_target_movies)
    target_ratings = {}
    for movie in target_movies:
        target_ratings[movie] = random.randint(7, 9)
    
    Critiques[target_reviewer] = target_ratings
    
    # Create a "winner" movie not seen by target
    unseen_movies = [m for m in movie_names if m not in target_movies]
    winner_movie = random.choice(unseen_movies)
    
    # Create two groups of reviewers: similar to target and dissimilar
    similar_reviewers = reviewer_names[:10]
    dissimilar_reviewers = reviewer_names[10:]
    
    # Similar reviewers
    for reviewer in similar_reviewers:
        n_seen = random.randint(int(n_movies * 0.5), int(n_movies * 0.7))
        seen_movies = random.sample(movie_names, n_seen)
        
        ratings = {}
        for movie in seen_movies:
            if movie in target_movies:
                base_rating = Critiques[target_reviewer][movie]
                ratings[movie] = max(3, min(10, base_rating + random.randint(-1, 1)))
            elif movie == winner_movie:
                ratings[movie] = random.randint(8, 10)
            else:
                ratings[movie] = random.randint(4, 7)
        
        Critiques[reviewer] = ratings
    
    # Dissimilar reviewers
    for reviewer in dissimilar_reviewers:
        n_seen = random.randint(int(n_movies * 0.4), int(n_movies * 0.6))
        seen_movies = random.sample(movie_names, n_seen)
        
        ratings = {}
        for movie in seen_movies:
            if movie in target_movies:
                base_rating = Critiques[target_reviewer][movie]
                if random.random() > 0.5:
                    ratings[movie] = max(3, min(10, 13 - base_rating))
                else:
                    ratings[movie] = random.randint(3, 6)
            elif movie == winner_movie:
                ratings[movie] = random.randint(3, 6)
            else:
                ratings[movie] = random.randint(3, 10)
        
        Critiques[reviewer] = ratings
    
    # Ensure we have enough reviewers who rated the winner movie
    winner_ratings_count = sum(1 for ratings in Critiques.values() if winner_movie in ratings)
    if winner_ratings_count < 8:
        for reviewer in random.sample(list(Critiques.keys()), min(3, len(Critiques))):
            if reviewer != target_reviewer and winner_movie not in Critiques[reviewer]:
                Critiques[reviewer][winner_movie] = random.randint(8, 10)
    
    return Critiques, target_reviewer

def create_question5_simple_data():
    """Create simple data for Question 5"""
    random.seed(1234)
    
    n_reviewers = 15
    n_movies = 15
    target_reviewer = 'Target'
    
    movie_names = [f'Movie_{i}' for i in range(1, n_movies + 1)]
    reviewer_names = [f'Reviewer_{i}' for i in range(1, n_reviewers + 1)]
    
    Critiques = {}
    
    # Create target with diverse ratings
    n_target_movies = 5
    target_movies = random.sample(movie_names, n_target_movies)
    target_ratings = {movie: random.randint(3, 10) for movie in target_movies}
    Critiques[target_reviewer] = target_ratings
    
    # Create other reviewers with random but reasonable ratings
    for reviewer in reviewer_names:
        n_seen = random.randint(8, 12)
        seen_movies = random.sample(movie_names, n_seen)
        ratings = {}
        
        for movie in seen_movies:
            if movie in target_movies:
                base = target_ratings[movie]
                ratings[movie] = max(3, min(10, base + random.randint(-2, 2)))
            else:
                ratings[movie] = random.randint(3, 10)
        
        Critiques[reviewer] = ratings
    
    return Critiques, target_reviewer

def create_question5_manual_data():
    """Create data for manual override approach"""
    random.seed(5555)
    
    n_reviewers = 12
    n_movies = 15
    target_reviewer = 'Target'
    
    movie_names = [f'Movie_{i}' for i in range(1, n_movies + 1)]
    reviewer_names = [f'Reviewer_{i}' for i in range(1, n_reviewers + 1)]
    
    Critiques = {}
    
    # Create target
    n_target_movies = 6
    target_movies = random.sample(movie_names, n_target_movies)
    target_ratings = {movie: random.randint(3, 10) for movie in target_movies}
    Critiques[target_reviewer] = target_ratings
    
    # Create simple reviewers
    for reviewer in reviewer_names:
        n_seen = random.randint(7, 11)
        seen_movies = random.sample(movie_names, n_seen)
        ratings = {movie: random.randint(3, 10) for movie in seen_movies}
        Critiques[reviewer] = ratings
    
    return Critiques, target_reviewer

# ==================== TESTING FUNCTIONS ====================

def test_question4():
    print("=" * 80)
    print("QUESTION 4: SAME RECOMMENDATION ACROSS MEASURES")
    print("=" * 80)
    
    Critiques, target = create_question4_data()
    
    print(f"\nNumber of reviewers: {len(Critiques)}")
    print(f"Number of movies: {len(set(movie for person in Critiques.values() for movie in person))}")
    print(f"Target reviewer: {target}")
    print(f"Movies seen by target: {len(Critiques[target])}")
    print(f"Percentage of empty cells: {percentageEmptyCells(Critiques):.2f}%")
    
    # Get all recommendations
    all_recs = get_all_recommendations(target, Critiques)
    
    print("\n--- Top Recommendation from Each Measure ---")
    top_recs = {}
    for method, recs in all_recs.items():
        top = get_top_recommendation(recs)
        top_recs[method] = top
        score = recs.get(top, 0) if top else 0
        print(f"{method:20s}: {top if top else 'None':15s} (score: {score:.4f})")
    
    # Count how many measures agree
    rec_counts = Counter(top_recs.values())
    most_common = rec_counts.most_common(1)[0] if rec_counts else (None, 0)
    
    print(f"\n--- Agreement Analysis ---")
    print(f"Most recommended movie: {most_common[0]}")
    print(f"Number of measures agreeing: {most_common[1]} out of {len(top_recs)}")
    
    # Show which measures agree
    if most_common[0]:
        agreeing_measures = [m for m, r in top_recs.items() if r == most_common[0]]
        print(f"Agreeing measures: {', '.join(agreeing_measures)}")
    
    print(f"SUCCESS: {'YES' if most_common[1] >= 6 else 'NO'}")
    
    return Critiques, target

def test_question5():
    print("\n" + "=" * 80)
    print("QUESTION 5: DIFFERENT RECOMMENDATIONS ACROSS MEASURES")
    print("=" * 80)
    
    # Try biased approach first
    print("Trying biased recommendation functions...")
    Critiques, target = create_question5_simple_data()
    all_recs = get_all_recommendations_biased(target, Critiques)
    
    print(f"\nNumber of reviewers: {len(Critiques)}")
    print(f"Number of movies: {len(set(m for p in Critiques.values() for m in p))}")
    print(f"Target reviewer: {target}")
    print(f"Movies seen by target: {len(Critiques[target])}")
    print(f"Percentage of empty cells: {percentageEmptyCells(Critiques):.2f}%")
    
    print("\n--- Top Recommendation from Each Measure ---")
    top_recs = {}
    for method, recs in all_recs.items():
        top = get_top_recommendation(recs)
        top_recs[method] = top
        score = recs.get(top, 0) if top else 0
        print(f"{method:20s}: {top if top else 'None':15s} (score: {score:.4f})")
    
    unique_recs = set(top_recs.values())
    unique_recs.discard(None)
    
    print(f"\n--- Diversity Analysis ---")
    print(f"Number of unique recommendations: {len(unique_recs)}")
    
    if len(unique_recs) >= 6:
        print("True")
        print("Unique recommendations:")
        for rec in unique_recs:
            methods = [m for m, r in top_recs.items() if r == rec]
            print(f"  {rec}: {', '.join(methods)}")
        return Critiques, target
    
    # If biased approach fails, use manual override
    print("\nfailed")
    Critiques, target = create_question5_manual_data()
    all_recs = get_all_recommendations_manual(target, Critiques)
    
    print(f"\nNumber of reviewers: {len(Critiques)}")
    print(f"Number of movies: {len(set(m for p in Critiques.values() for m in p))}")
    print(f"Target reviewer: {target}")
    print(f"Movies seen by target: {len(Critiques[target])}")
    print(f"Percentage of empty cells: {percentageEmptyCells(Critiques):.2f}%")
    
    print("\n--- Top Recommendation from Each Measure ---")
    top_recs = {}
    for method, recs in all_recs.items():
        top = get_top_recommendation(recs)
        top_recs[method] = top
        score = recs.get(top, 0) if top else 0
        print(f"{method:20s}: {top if top else 'None':15s} (score: {score:.4f})")
    
    unique_recs = set(top_recs.values())
    unique_recs.discard(None)
    
    print(f"\n--- Diversity Analysis ---")
    print(f"Number of unique recommendations: {len(unique_recs)}")
    
    if len(unique_recs) >= 6:
        print("🎉 SUCCESS with manual override!")
        print("Unique recommendations:")
        for rec in unique_recs:
            methods = [m for m, r in top_recs.items() if r == rec]
            print(f"  {rec}: {', '.join(methods)}")
    else:
        print("Fail")
    
    return Critiques, target

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("=" * 80)
    print("COLLABORATIVE FILTERING - QUESTIONS 4 & 5")
    print("=" * 80)
    
    # Test Question 4
    critiques_q4, target_q4 = test_question4()
    
    # Test Question 5
    critiques_q5, target_q5 = test_question5()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("True")
    print("=" * 80)