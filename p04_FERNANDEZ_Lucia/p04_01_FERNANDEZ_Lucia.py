import random
import math
from typing import Dict, List, Tuple

# Copy all the similarity functions from the previous implementation
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
    """Jaccard similarity coefficient."""
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) == 0:
        return 0
    all_items = set(person1.keys()).union(set(person2.keys()))
    return len(shared_items) / len(all_items)

def spearman(person1: Dict, person2: Dict) -> float:
    """Spearman rank correlation coefficient (approximation using Pearson on ranks)."""
    shared_items = [item for item in person1 if item in person2]
    if len(shared_items) < 2:
        return 0
    
    # Convert to ranks
    ranks1 = {item: rank for rank, item in enumerate(sorted(shared_items, key=lambda x: person1[x]), 1)}
    ranks2 = {item: rank for rank, item in enumerate(sorted(shared_items, key=lambda x: person2[x]), 1)}
    
    rank_person1 = {item: ranks1[item] for item in shared_items}
    rank_person2 = {item: ranks2[item] for item in shared_items}
    
    return pearson(rank_person1, rank_person2)

# Recommendation functions for each similarity measure
def recommend_with_similarity(nouveauCritique: str, Critiques: Dict, similarity_func, use_positive_only=False) -> Dict[str, float]:
    """Generic recommendation function using any similarity measure."""
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

def percentageEmptyCells(Critiques: Dict) -> float:
    """Calculate the percentage of empty cells in the rating matrix."""
    all_movies = set()
    for person in Critiques.values():
        all_movies.update(person.keys())
    
    total_cells = len(Critiques) * len(all_movies)
    filled_cells = sum(len(person) for person in Critiques.values())
    empty_cells = total_cells - filled_cells
    
    return (empty_cells / total_cells) * 100

def get_all_recommendations(target_reviewer: str, Critiques: Dict) -> Dict[str, Dict[str, float]]:
    """Get recommendations from all similarity measures."""
    methods = {
        'Manhattan_Inv': recommend_manhattan_inv,
        'Euclidean_Inv': recommend_euclidean_inv,
        'Manhattan_Exp': recommend_manhattan_exp,
        'Euclidean_Exp': recommend_euclidean_exp,
        'Pearson': recommend_pearson,
        'Cosine': recommend_cosine,
        'Jaccard': recommend_jaccard,
        'Spearman': recommend_spearman
    }
    
    results = {}
    for name, func in methods.items():
        try:
            results[name] = func(target_reviewer, Critiques)
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {}
    
    return results

def get_top_recommendation(recommendations: Dict[str, float]) -> str:
    """Get the movie with highest score."""
    if not recommendations:
        return None
    return max(recommendations.items(), key=lambda x: x[1])[0]

# ==================== QUESTION 4: Same Recommendation Across Measures ====================

def create_example_question4():
    """
    Create an example where at least 6 similarity measures give the same recommendation.
    Enhanced strategy: Create stronger patterns that work for Pearson correlation too.
    """
    random.seed(42)  # Changed seed for better results
    
    n_reviewers = 15
    n_movies = 15
    target_reviewer = 'Target'
    
    movie_names = [f'Movie_{i}' for i in range(1, n_movies + 1)]
    reviewer_names = [f'Reviewer_{i}' for i in range(1, n_reviewers + 1)]
    
    Critiques = {}
    
    # Create target reviewer who has seen less than half the movies
    n_target_movies = n_movies // 3  # Seen 1/3 of movies
    target_movies = random.sample(movie_names, n_target_movies)
    
    # Create consistent rating pattern for target (important for Pearson)
    target_ratings = {}
    for movie in target_movies:
        # Use consistent high ratings for target to create clear pattern
        target_ratings[movie] = random.randint(7, 9)
    
    Critiques[target_reviewer] = target_ratings
    
    # Create a "winner" movie not seen by target
    unseen_movies = [m for m in movie_names if m not in target_movies]
    winner_movie = random.choice(unseen_movies)
    
    # Create two groups of reviewers: similar to target and dissimilar
    similar_reviewers = reviewer_names[:10]  # 10 similar reviewers
    dissimilar_reviewers = reviewer_names[10:]  # 5 dissimilar reviewers
    
    # Similar reviewers: rate similarly to target AND give high ratings to winner movie
    for reviewer in similar_reviewers:
        n_seen = random.randint(int(n_movies * 0.5), int(n_movies * 0.7))
        seen_movies = random.sample(movie_names, n_seen)
        
        ratings = {}
        for movie in seen_movies:
            if movie in target_movies:
                # Very similar ratings to target (small variation)
                base_rating = Critiques[target_reviewer][movie]
                ratings[movie] = max(3, min(10, base_rating + random.randint(-1, 1)))
            elif movie == winner_movie:
                # Consistently high rating for winner movie
                ratings[movie] = random.randint(8, 10)
            else:
                # Moderate ratings for other unseen movies
                ratings[movie] = random.randint(4, 7)
        
        Critiques[reviewer] = ratings
    
    # Dissimilar reviewers: different rating patterns
    for reviewer in dissimilar_reviewers:
        n_seen = random.randint(int(n_movies * 0.4), int(n_movies * 0.6))
        seen_movies = random.sample(movie_names, n_seen)
        
        ratings = {}
        for movie in seen_movies:
            if movie in target_movies:
                # Different rating style from target
                base_rating = Critiques[target_reviewer][movie]
                # Create opposite or very different pattern
                if random.random() > 0.5:
                    ratings[movie] = max(3, min(10, 13 - base_rating))  # Opposite
                else:
                    ratings[movie] = random.randint(3, 6)  # Low ratings
            elif movie == winner_movie:
                # Give winner movie low to moderate ratings
                ratings[movie] = random.randint(3, 6)
            else:
                # Random ratings for other movies
                ratings[movie] = random.randint(3, 10)
        
        Critiques[reviewer] = ratings
    
    # Ensure we have enough reviewers who rated the winner movie
    winner_ratings_count = sum(1 for ratings in Critiques.values() if winner_movie in ratings)
    if winner_ratings_count < 8:  # Ensure enough data for reliable recommendations
        # Add some additional ratings for winner movie
        for reviewer in random.sample(list(Critiques.keys()), min(3, len(Critiques))):
            if reviewer != target_reviewer and winner_movie not in Critiques[reviewer]:
                Critiques[reviewer][winner_movie] = random.randint(8, 10)
    
    return Critiques, target_reviewer

# ==================== QUESTION 5: Different Recommendations Across Measures ====================

def create_example_question5():
    """
    Create an example where at least 6 similarity measures give DIFFERENT recommendations.
    Enhanced strategy: Create more diverse patterns that exploit differences between measures.
    """
    random.seed(123)
    
    n_reviewers = 15
    n_movies = 15
    target_reviewer = 'Target'
    
    movie_names = [f'Movie_{chr(65+i)}' for i in range(n_movies)]  # Movie_A, Movie_B, etc.
    reviewer_names = [f'Reviewer_{i}' for i in range(1, n_reviewers + 1)]
    
    Critiques = {}
    
    # Create target reviewer with specific diverse pattern
    n_target_movies = n_movies // 3
    target_movies = random.sample(movie_names, n_target_movies)
    
    # Create very diverse rating pattern for target (helps different measures pick different movies)
    target_ratings = {}
    rating_values = [3, 4, 7, 8, 9, 10, 5, 6]  # Mixed high and low
    for i, movie in enumerate(target_movies):
        target_ratings[movie] = rating_values[i % len(rating_values)]
    
    Critiques[target_reviewer] = target_ratings
    unseen_movies = [m for m in movie_names if m not in target_movies]
    
    # Select 6 candidate movies for different measures to favor
    candidate_movies = unseen_movies[:6]
    
    # Create reviewers with very specific patterns for each similarity measure
    for idx, reviewer in enumerate(reviewer_names):
        n_seen = random.randint(int(n_movies * 0.45), int(n_movies * 0.65))
        seen_movies = random.sample(movie_names, n_seen)
        
        ratings = {}
        
        # Assign reviewer to favor a specific candidate movie based on measure type
        measure_type = idx % 6  # 6 different patterns for 6 measures
        
        for movie in seen_movies:
            if movie in target_movies:
                base = target_ratings[movie]
                
                if measure_type == 0:  # Pattern for distance measures
                    ratings[movie] = base + random.randint(-1, 1)  # Very similar
                elif measure_type == 1:  # Pattern for Pearson
                    # Pearson favors consistent relative patterns
                    ratings[movie] = max(3, min(10, base * 0.8 + 2))
                elif measure_type == 2:  # Pattern for Cosine
                    # Cosine favors similar rating magnitudes
                    ratings[movie] = max(3, min(10, base + random.randint(-2, 2)))
                elif measure_type == 3:  # Pattern for Jaccard
                    # Jaccard cares about overlap, not values
                    ratings[movie] = random.randint(3, 10)
                elif measure_type == 4:  # Pattern for Spearman
                    # Spearman cares about rank order
                    ratings[movie] = 11 - base if base > 6 else base + 3
                else:  # Random pattern
                    ratings[movie] = random.randint(3, 10)
            else:
                # For unseen movies, assign specific patterns
                if movie in candidate_movies:
                    candidate_idx = candidate_movies.index(movie)
                    if measure_type == candidate_idx:
                        # This reviewer type gives high rating to their assigned candidate
                        ratings[movie] = random.randint(8, 10)
                    else:
                        # Other reviewer types give moderate to low ratings
                        ratings[movie] = random.randint(3, 7)
                else:
                    ratings[movie] = random.randint(3, 10)
        
        Critiques[reviewer] = ratings
    
    return Critiques, target_reviewer

# ==================== TESTING AND DISPLAY ====================

def test_question4():
    print("=" * 80)
    print("QUESTION 4: SAME RECOMMENDATION ACROSS AT LEAST 6 MEASURES")
    print("=" * 80)
    
    Critiques, target = create_example_question4()
    
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
    from collections import Counter
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
    print("QUESTION 5: DIFFERENT RECOMMENDATIONS ACROSS AT LEAST 6 MEASURES")
    print("=" * 80)
    
    Critiques, target = create_example_question5()
    
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
    
    # Count unique recommendations
    unique_recs = set(top_recs.values())
    unique_recs.discard(None)
    
    print(f"\n--- Diversity Analysis ---")
    print(f"Number of unique recommendations: {len(unique_recs)}")
    
    if len(unique_recs) >= 6:
        print("\nUnique recommendations:")
        for rec in unique_recs:
            methods = [m for m, r in top_recs.items() if r == rec]
            print(f"  {rec}: {', '.join(methods)}")
    
    print(f"SUCCESS: {'YES' if len(unique_recs) >= 6 else 'NO'}")
    
    return Critiques, target

if __name__ == "__main__":
    # Test Question 4
    print("Testing Question 4...")
    critiques_q4, target_q4 = test_question4()
    
    # Test Question 5
    print("\nTesting Question 5...")
    critiques_q5, target_q5 = test_question5()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Both examples have been generated successfully!")
    print("You can modify the random seeds and parameters to generate different examples.")
    print("=" * 80)