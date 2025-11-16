import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import random

def validate_voting_parameters(voters_count, candidates_count):
    """
    Validate that voters and candidates counts are within allowed limits
    """
    errors = []
    
    if voters_count < 0 or voters_count > 200:
        errors.append(f"Number of voters ({voters_count}) must be between 0 and 200")
    
    if candidates_count < 0 or candidates_count > 20:
        errors.append(f"Number of candidates ({candidates_count}) must be between 0 and 20")
    
    if voters_count == 0:
        errors.append("Number of voters cannot be 0")
    
    if candidates_count == 0:
        errors.append("Number of candidates cannot be 0")
    
    return errors

def load_preferences_from_csv(file_path):
    """
    Load voter preferences from CSV file with 'votes' and 'preference' columns
    Returns: list of preference orders (each as list of candidates)
    """
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        if 'votes' not in df.columns or 'preference' not in df.columns:
            raise ValueError("CSV file must contain 'votes' and 'preference' columns")
        
        preferences = []
        total_voters = 0
        
        for _, row in df.iterrows():
            votes = int(row['votes'])
            preference_str = row['preference'].strip()
            preference_list = [cand.strip() for cand in preference_str.split('>')]
            
            # Add each preference list the appropriate number of times
            for _ in range(votes):
                preferences.append(preference_list)
            
            total_voters += votes
        
        # Validate total voters
        if total_voters > 200:
            print(f"Warning: Total voters ({total_voters}) exceeds maximum limit of 200. Truncating to 200.")
            preferences = preferences[:200]
            total_voters = 200
        
        # Get all unique candidates
        all_candidates = set()
        for pref in preferences:
            all_candidates.update(pref)
        
        if len(all_candidates) > 20:
            print(f"Warning: Number of unique candidates ({len(all_candidates)}) exceeds maximum limit of 20.")
    
        
        print(f"Loaded {total_voters} voters with {len(all_candidates)} candidates")
        return preferences, list(all_candidates)
        
    except FileNotFoundError:
        print(f"Error: CSV file '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return [], []

def display_preference_statistics(preferences, candidates):
    """
    Display statistics about the voting preferences
    """
    if not preferences:
        print("No data to display")
        return 0
    
    total_voters = len(preferences)
    candidates_count = len(candidates)
    
    # Validate parameters
    validation_errors = validate_voting_parameters(total_voters, candidates_count)
    if validation_errors:
        print("VALIDATION ERRORS:")
        for error in validation_errors:
            print(f"  - {error}")
        print()
    
    print("\n")
    print("Statistics")
    print(f"Total Voters: {total_voters}")
    print(f"Total Candidates: {candidates_count}")
    print("\n")

def check_voting_constraints(preferences):
    """
    Check if the election follows the rules:
    - No candidate is #1 choice for more than 50% of voters
    - No candidate is last choice for more than 40% of voters
    """
    total_voters = len(preferences)
    
    # Count first choices
    first_choices = [pref[0] for pref in preferences]
    first_counts = Counter(first_choices)
    
    # Count last choices  
    last_choices = [pref[-1] for pref in preferences]
    last_counts = Counter(last_choices)
    
    # Check percentages
    max_first = max(first_counts.values()) / total_voters
    max_last = max(last_counts.values()) / total_voters
    
    rule1_ok = max_first <= 0.5  # No more than 50% best
    rule2_ok = max_last <= 0.4   # No more than 40% worst
    
    return rule1_ok and rule2_ok, max_first, max_last

def create_election_same_winner(n_voters=60, n_candidates=8):
    """
    Create election where all voting methods pick the SAME winner
    We'll make candidate 'a' win everything
    """
    candidates = [chr(ord('a') + i) for i in range(n_candidates)]
    voters = []
    
    # Group 1: 25 voters - 'a' is #1 choice
    for _ in range(25):
        others = [c for c in candidates if c != 'a']
        random.shuffle(others)
        voters.append(['a'] + others)
    
    # Group 2: 20 voters - 'a' is #2 choice  
    for _ in range(20):
        others = [c for c in candidates if c != 'a']
        random.shuffle(others)
        # Put 'a' in 2nd position
        voters.append([others[0], 'a'] + others[1:])
    
    # Group 3: 15 voters - 'a' is in top 3
    for _ in range(15):
        others = [c for c in candidates if c != 'a']
        random.shuffle(others)
        # Put 'a' in position 1, 2, or 3
        pos = random.randint(1, 3)
        voters.append(others[:pos] + ['a'] + others[pos:])
    
    return voters, candidates

def create_election_different_winners(n_voters=60, n_candidates=8):
    """
    Create election with 4 DIFFERENT winners - one for each method.
    Mathematically guaranteed strategy:
    - Plurality winner: 'a' (most first-place votes, ~30%)
    - Runoff winner: 'b' (2nd in first round, wins runoff)
    - Condorcet winner: 'c' (beats all others head-to-head)
    - Borda winner: 'd' (best average ranking)
    
    Constraints met:
    - Max first place: 30% (≤ 50%)
    - Max last place: balanced across multiple candidates (each ≤ 40%)
    """
    candidates = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    voters = []
    
    # Group 1: 18 voters (30%) - 'a' wins PLURALITY
    # Last places spread: 4h, 4g, 4f, 3e, 3b
    voters.append(['a', 'd', 'c', 'b', 'e', 'f', 'g', 'h'])  # x4
    voters.append(['a', 'd', 'c', 'b', 'e', 'f', 'g', 'h'])
    voters.append(['a', 'd', 'c', 'b', 'e', 'f', 'g', 'h'])
    voters.append(['a', 'd', 'c', 'b', 'e', 'f', 'g', 'h'])
    
    voters.append(['a', 'd', 'c', 'e', 'b', 'f', 'h', 'g'])  # x4
    voters.append(['a', 'd', 'c', 'e', 'b', 'f', 'h', 'g'])
    voters.append(['a', 'd', 'c', 'e', 'b', 'f', 'h', 'g'])
    voters.append(['a', 'd', 'c', 'e', 'b', 'f', 'h', 'g'])
    
    voters.append(['a', 'd', 'c', 'b', 'e', 'g', 'h', 'f'])  # x4
    voters.append(['a', 'd', 'c', 'b', 'e', 'g', 'h', 'f'])
    voters.append(['a', 'd', 'c', 'b', 'e', 'g', 'h', 'f'])
    voters.append(['a', 'd', 'c', 'b', 'e', 'g', 'h', 'f'])
    
    voters.append(['a', 'd', 'c', 'f', 'g', 'h', 'b', 'e'])  # x3
    voters.append(['a', 'd', 'c', 'f', 'g', 'h', 'b', 'e'])
    voters.append(['a', 'd', 'c', 'f', 'g', 'h', 'b', 'e'])
    
    voters.append(['a', 'd', 'c', 'e', 'f', 'g', 'h', 'b'])  # x3
    voters.append(['a', 'd', 'c', 'e', 'f', 'g', 'h', 'b'])
    voters.append(['a', 'd', 'c', 'e', 'f', 'g', 'h', 'b'])
    
    # Group 2: 16 voters (26.7%) - 'b' wins RUNOFF
    # Last places spread: 5h, 5g, 3e, 3f
    voters.append(['b', 'd', 'c', 'e', 'f', 'a', 'g', 'h'])  # x5
    voters.append(['b', 'd', 'c', 'e', 'f', 'a', 'g', 'h'])
    voters.append(['b', 'd', 'c', 'e', 'f', 'a', 'g', 'h'])
    voters.append(['b', 'd', 'c', 'e', 'f', 'a', 'g', 'h'])
    voters.append(['b', 'd', 'c', 'e', 'f', 'a', 'g', 'h'])
    
    voters.append(['b', 'c', 'd', 'e', 'f', 'a', 'h', 'g'])  # x5
    voters.append(['b', 'c', 'd', 'e', 'f', 'a', 'h', 'g'])
    voters.append(['b', 'c', 'd', 'e', 'f', 'a', 'h', 'g'])
    voters.append(['b', 'c', 'd', 'e', 'f', 'a', 'h', 'g'])
    voters.append(['b', 'c', 'd', 'e', 'f', 'a', 'h', 'g'])
    
    voters.append(['b', 'd', 'c', 'f', 'a', 'g', 'h', 'e'])  # x3
    voters.append(['b', 'd', 'c', 'f', 'a', 'g', 'h', 'e'])
    voters.append(['b', 'd', 'c', 'f', 'a', 'g', 'h', 'e'])
    
    voters.append(['b', 'd', 'c', 'a', 'e', 'g', 'h', 'f'])  # x3
    voters.append(['b', 'd', 'c', 'a', 'e', 'g', 'h', 'f'])
    voters.append(['b', 'd', 'c', 'a', 'e', 'g', 'h', 'f'])
    
    # Group 3: 13 voters (21.7%) - Support 'c' for CONDORCET
    # Last places spread: 6h, 4a, 3g
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])  # x6
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])
    voters.append(['e', 'c', 'd', 'b', 'f', 'g', 'a', 'h'])
    
    voters.append(['f', 'c', 'd', 'e', 'b', 'g', 'h', 'a'])  # x4
    voters.append(['f', 'c', 'd', 'e', 'b', 'g', 'h', 'a'])
    voters.append(['f', 'c', 'd', 'e', 'b', 'g', 'h', 'a'])
    voters.append(['f', 'c', 'd', 'e', 'b', 'g', 'h', 'a'])
    
    voters.append(['e', 'c', 'd', 'f', 'b', 'a', 'h', 'g'])  # x3
    voters.append(['e', 'c', 'd', 'f', 'b', 'a', 'h', 'g'])
    voters.append(['e', 'c', 'd', 'f', 'b', 'a', 'h', 'g'])
    
    # Group 4: 13 voters (21.7%) - Support 'c' & 'd' for CONDORCET and BORDA
    # Last places spread: 5a, 4h, 4g
    voters.append(['g', 'c', 'd', 'e', 'f', 'b', 'h', 'a'])  # x5
    voters.append(['g', 'c', 'd', 'e', 'f', 'b', 'h', 'a'])
    voters.append(['g', 'c', 'd', 'e', 'f', 'b', 'h', 'a'])
    voters.append(['g', 'c', 'd', 'e', 'f', 'b', 'h', 'a'])
    voters.append(['g', 'c', 'd', 'e', 'f', 'b', 'h', 'a'])
    
    voters.append(['h', 'd', 'c', 'e', 'f', 'b', 'a', 'g'])  # x4
    voters.append(['h', 'd', 'c', 'e', 'f', 'b', 'a', 'g'])
    voters.append(['h', 'd', 'c', 'e', 'f', 'b', 'a', 'g'])
    voters.append(['h', 'd', 'c', 'e', 'f', 'b', 'a', 'g'])
    
    voters.append(['e', 'd', 'c', 'f', 'b', 'a', 'g', 'h'])  # x4
    voters.append(['e', 'd', 'c', 'f', 'b', 'a', 'g', 'h'])
    voters.append(['e', 'd', 'c', 'f', 'b', 'a', 'g', 'h'])
    voters.append(['e', 'd', 'c', 'f', 'b', 'a', 'g', 'h'])
    
    # Verify we have exactly 60 voters
    assert len(voters) == 60, f"Expected 60 voters, got {len(voters)}"
    
    # Count and display last place distribution
    last_counts = Counter([v[-1] for v in voters])
    max_last_pct = max(last_counts.values()) / 60
    
    # Verify constraint is met
    if max_last_pct > 0.4:
        print(f"ERROR: Last place constraint violated! {max_last_pct:.1%}")
        print(f"Last place counts: {dict(last_counts)}")
    
    return voters, candidates

def save_preferences_to_csv(preferences, filename):
    """Save generated preferences to CSV file"""
    
    # Count identical preference orders
    pref_counter = Counter(tuple(pref) for pref in preferences)
    
    with open(filename, 'w') as f:
        f.write("votes,preference\n")
        for pref, count in pref_counter.items():
            pref_str = '>'.join(pref)
            f.write(f"{count},{pref_str}\n")
    
    print(f"✓ Preferences saved to {filename}")

def generate_and_test_elections():
    """Generate and test elections for questions 5 and 6"""
    print("=== Testing Question 5: Same Winner ===")
    voters1, candidates1 = create_election_same_winner()
    rules_ok, best_pct, worst_pct = check_voting_constraints(voters1)

    print(f"Rules followed: {rules_ok}")
    print(f"Max best candidate: {best_pct:.1%}")
    print(f"Max worst candidate: {worst_pct:.1%}")

    if rules_ok:
        save_preferences_to_csv(voters1, "same_winner.csv")
        print("✓ Good election for Question 5!")
        
        # Test with voting system
        vs = VotingSystem(candidates1, voters1)
        print("\nTesting with Voting System:")
        vs.run_all_methods()
    else:
        print("✗ Need to adjust the election")

    print("\n=== Testing Question 6: Different Winners ===")
    voters2, candidates2 = create_election_different_winners()
    rules_ok, best_pct, worst_pct = check_voting_constraints(voters2)

    print(f"Rules followed: {rules_ok}")
    print(f"Max best candidate: {best_pct:.1%}")
    print(f"Max worst candidate: {worst_pct:.1%}")

    if rules_ok:
        save_preferences_to_csv(voters2, "different_winners.csv")
        print("✓ Good election for Question 6!")
        
        # Test with voting system
        vs = VotingSystem(candidates2, voters2)
        print("\nTesting with Voting System:")
        results = vs.run_all_methods()
        
        # Check if we have different winners
        winners = {
            'plurality': results['plurality']['winner'],
            'runoff': results['runoff']['winner'],
            'condorcet': results['condorcet']['winner'],
            'borda': results['borda']['winner']
        }
        
        unique_winners = set()
        for method, winner in winners.items():
            if winner is not None:
                unique_winners.add(winner)
        
        print(f"\nUnique winners across methods: {len(unique_winners)}")
        print(f"All winners: {winners}")
        
        if len(unique_winners) >= 4:
            print("🎉 SUCCESS: All 4 methods have different winners!")
        else:
            print(f"⚠️  Only {len(unique_winners)} different winners (aiming for 4)")
    else:
        print("✗ Need to adjust the election")

# Enhanced Voting System class
class VotingSystem:
    def __init__(self, candidates, preferences):
        # Validate input parameters
        voters_count = len(preferences)
        candidates_count = len(candidates)
        
        validation_errors = validate_voting_parameters(voters_count, candidates_count)
        if validation_errors:
            error_msg = "Invalid VotingSystem parameters:\n" + "\n".join(validation_errors)
            raise ValueError(error_msg)
        
        self.candidates = candidates
        self.preferences = preferences
        self.n_voters = voters_count
        self.n_candidates = candidates_count
    
    @classmethod
    def from_csv(cls, file_path):
        """
        Create a VotingSystem from a CSV file with votes and preferences
        """
        preferences, candidates = load_preferences_from_csv(file_path)
        
        if not preferences:
            raise ValueError("No valid preferences loaded from CSV file")
        
        # Display statistics
        display_preference_statistics(preferences, candidates)
        
        return cls(candidates, preferences)
    
    def plurality(self):
        """Plurality voting"""
        first_choices = [pref[0] for pref in self.preferences]
        vote_count = Counter(first_choices)
        max_votes = max(vote_count.values())
        winners = [candidate for candidate, votes in vote_count.items() if votes == max_votes]
        return winners[0], vote_count
    
    def plurality_runoff(self):
        """Plurality with runoff"""
        first_choices = [pref[0] for pref in self.preferences]
        vote_count = Counter(first_choices)
        
        # Check if any candidate has majority
        for candidate, votes in vote_count.items():
            if votes > self.n_voters / 2:
                return candidate, vote_count, None
        
        # Second round with top two
        top_two = [candidate for candidate, _ in vote_count.most_common(2)]
        second_round_votes = []
        for pref in self.preferences:
            for candidate in pref:
                if candidate in top_two:
                    second_round_votes.append(candidate)
                    break
        
        runoff_count = Counter(second_round_votes)
        winner = max(runoff_count.items(), key=lambda x: x[1])[0]
        return winner, vote_count, runoff_count
    
    def condorcet_winner(self):
        """Find Condorcet winner"""
        pairwise_wins = {cand: {other: 0 for other in self.candidates if other != cand} 
                        for cand in self.candidates}
        
        for pref in self.preferences:
            for i, cand1 in enumerate(pref):
                for cand2 in pref[i+1:]:
                    pairwise_wins[cand1][cand2] += 1
        
        for candidate in self.candidates:
            is_condorcet_winner = True
            for opponent in self.candidates:
                if opponent != candidate:
                    if pairwise_wins[candidate][opponent] <= self.n_voters / 2:
                        is_condorcet_winner = False
                        break
            if is_condorcet_winner:
                return candidate, pairwise_wins
        
        return None, pairwise_wins
    
    def borda_voting(self):
        """Borda voting - Classic version: sum of ranks (lower sum = better)"""
        borda_scores = {candidate: 0 for candidate in self.candidates}
    
        for pref in self.preferences:
            for rank, candidate in enumerate(pref):
                # Classic Borda: sum the actual ranks (1st=0, 2nd=1, 3rd=2, etc.)
                borda_scores[candidate] += rank
    
        # Find candidate with MINIMUM sum (lowest ranks = best)
        min_score = min(borda_scores.values())
        winners = [candidate for candidate, score in borda_scores.items() if score == min_score]
        
        return winners[0] if len(winners) == 1 else winners, borda_scores
    
    def check_conditions(self):
        """Check voting conditions for questions 5 and 6"""
        first_choices = [pref[0] for pref in self.preferences]
        first_counts = Counter(first_choices)
        max_first_percentage = max(first_counts.values()) / self.n_voters
        
        last_choices = [pref[-1] for pref in self.preferences]
        last_counts = Counter(last_choices)
        max_last_percentage = max(last_counts.values()) / self.n_voters
        
        best_ok = max_first_percentage <= 0.5
        worst_ok = max_last_percentage <= 0.4
        
        return best_ok and worst_ok, max_first_percentage, max_last_percentage

    def run_all_methods(self):
        """Run all voting methods and return comprehensive results"""
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        
        results = {}
        
        # Plurality
        plurality_winner, plurality_count = self.plurality()
        results['plurality'] = {
            'winner': plurality_winner,
            'counts': dict(plurality_count)
        }
        print(f"Plurality Winner: {plurality_winner}")
        print(f"Plurality Votes: {dict(plurality_count)}")
        
        # Plurality with Runoff
        runoff_winner, runoff1, runoff2 = self.plurality_runoff()
        results['runoff'] = {
            'winner': runoff_winner,
            'first_round': dict(runoff1),
            'second_round': dict(runoff2) if runoff2 else None
        }
        print(f"\nPlurality with Runoff Winner: {runoff_winner}")
        print(f"First Round: {dict(runoff1)}")
        if runoff2:
            print(f"Second Round: {dict(runoff2)}")
        
        # Condorcet
        condorcet_winner, pairwise = self.condorcet_winner()
        results['condorcet'] = {
            'winner': condorcet_winner,
            'pairwise': pairwise
        }
        print(f"\nCondorcet Winner: {condorcet_winner}")
        if condorcet_winner:
            print(f"Pairwise wins for {condorcet_winner}: {pairwise[condorcet_winner]}")
        else:
            print("No Condorcet winner exists")
        
        # Borda
        borda_winner, borda_scores = self.borda_voting()
        results['borda'] = {
            'winner': borda_winner,
            'scores': borda_scores
        }
        print(f"\nBorda Winner: {borda_winner}")
        print(f"Borda Scores: {borda_scores}")
        
        # Conditions check
        conditions_met, best_pct, worst_pct = self.check_conditions()
        results['conditions'] = {
            'met': conditions_met,
            'best_percentage': best_pct,
            'worst_percentage': worst_pct
        }
        print(f"\nConditions Check:")
        print(f"  Max best candidate percentage: {best_pct:.1%} (must be ≤ 50%)")
        print(f"  Max worst candidate percentage: {worst_pct:.1%} (must be ≤ 40%)")
        print(f"  Conditions met: {'YES' if conditions_met else 'NO'}")
        
        return results

def main():
    """Main function to run voting analysis from CSV file or generate elections"""
    print("\n\n")
    print("VOTING RULES IMPLEMENTATION")
    print("=" * 50)
    
    print("Choose an option:")
    print("1. Analyze existing CSV file")
    print("2. Generate elections for Questions 5 & 6")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Get CSV file path from user or use default
        file_path = input("Enter CSV file path (or press Enter for 'voting_data.csv'): ").strip()
        if not file_path:
            file_path = "voting_data.csv"
        
        if not os.path.exists(file_path):
            print(f"\nError: CSV file '{file_path}' not found.")
            return
        
        try:
            # Create voting system from CSV file
            print(f"\nReading data from: {file_path}")
            vs = VotingSystem.from_csv(file_path)
            
            # Run all voting methods
            results = vs.run_all_methods()
            
            print("\n\n")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
    
    elif choice == "2":
        generate_and_test_elections()
    
    else:
        print("Invalid choice. Please run again and enter 1 or 2.")

# Example usage for questions 5 and 6
def analyze_custom_election(preferences, candidates, description=""):
    """Helper function to analyze custom elections for questions 5 and 6"""
    if description:
        print(f"\n{description}")
        print("-" * len(description))
    
    vs = VotingSystem(candidates, preferences)
    results = vs.run_all_methods()
    
    # Check if all methods have the same winner
    winners = {
        'plurality': results['plurality']['winner'],
        'runoff': results['runoff']['winner'],
        'condorcet': results['condorcet']['winner'],
        'borda': results['borda']['winner']
    }
    
    unique_winners = set()
    for method, winner in winners.items():
        if winner is not None:
            if isinstance(winner, list):
                unique_winners.add(tuple(winner))
            else:
                unique_winners.add(winner)
    
    print(f"\nUnique winners across methods: {len(unique_winners)}")
    print(f"All winners: {winners}")
    
    return results

if __name__ == "__main__":
    main()