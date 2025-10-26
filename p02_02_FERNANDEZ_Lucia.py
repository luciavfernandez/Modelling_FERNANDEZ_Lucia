import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pulp
import numpy as np
import pandas as pd
from itertools import combinations

class ParisTouristOptimizer:
    def __init__(self):
        # Tourist site data
        self.sites = {
            'TE': {'name': 'La Tour Eiffel', 'duration': 4.5, 'price': 16.50, 'appreciation': 5},
            'ML': {'name': 'Le Musée du Louvre', 'duration': 3, 'price': 14, 'appreciation': 4},
            'AT': {'name': 'L\'Arc de triomphe', 'duration': 1, 'price': 10.50, 'appreciation': 3},
            'MO': {'name': 'Le Musée d\'Orsay', 'duration': 2, 'price': 11, 'appreciation': 2},
            'JT': {'name': 'Le Jardin des tuileries', 'duration': 1.5, 'price': 0, 'appreciation': 3},
            'CA': {'name': 'Les Catacombes', 'duration': 2, 'price': 10, 'appreciation': 4},
            'CP': {'name': 'Le Centre Pompidou', 'duration': 2.5, 'price': 10, 'appreciation': 1},
            'CN': {'name': 'La Cathédrale Notre Dame', 'duration': 2, 'price': 7, 'appreciation': 5},
            'BS': {'name': 'La Basilique du Sacré-Coeur', 'duration': 2, 'price': 10, 'appreciation': 4},
            'SC': {'name': 'La Sainte Chapelle', 'duration': 1.5, 'price': 8.50, 'appreciation': 1},
            'PC': {'name': 'La Place de la Concorde', 'duration': 0.75, 'price': 0, 'appreciation': 3},
            'TM': {'name': 'La Tour Montparnasse', 'duration': 2, 'price': 12, 'appreciation': 2},
            'AC': {'name': 'L\'Avenue des Champs-Elysées', 'duration': 1.5, 'price': 0, 'appreciation': 5}
        }
        
        # Distance matrix (walking distance in km)
        self.distances = {
            'TE': {'TE': 0, 'ML': 3.8, 'AT': 2.1, 'MO': 2.4, 'JT': 3.5, 'CA': 4.2, 'CP': 5.0, 'CN': 4.4, 'BS': 5.5, 'SC': 4.2, 'PC': 2.5, 'TM': 3.1, 'AC': 1.9},
            'ML': {'TE': 3.8, 'ML': 0, 'AT': 3.8, 'MO': 1.1, 'JT': 1.3, 'CA': 3.3, 'CP': 1.3, 'CN': 1.1, 'BS': 3.4, 'SC': 0.8, 'PC': 1.7, 'TM': 2.5, 'AC': 2.8},
            'AT': {'TE': 2.1, 'ML': 3.8, 'AT': 0, 'MO': 3.1, 'JT': 3.0, 'CA': 5.8, 'CP': 4.8, 'CN': 4.9, 'BS': 4.3, 'SC': 4.6, 'PC': 2.2, 'TM': 4.4, 'AC': 1.0},
            'MO': {'TE': 2.4, 'ML': 1.1, 'AT': 3.1, 'MO': 0, 'JT': 0.9, 'CA': 3.1, 'CP': 2.5, 'CN': 2.0, 'BS': 3.9, 'SC': 1.8, 'PC': 1.0, 'TM': 2.3, 'AC': 2.1},
            'JT': {'TE': 3.5, 'ML': 1.3, 'AT': 3.0, 'MO': 0.9, 'JT': 0, 'CA': 4.2, 'CP': 2.0, 'CN': 2.4, 'BS': 2.7, 'SC': 2.0, 'PC': 1.0, 'TM': 3.4, 'AC': 2.1},
            'CA': {'TE': 4.2, 'ML': 3.3, 'AT': 5.8, 'MO': 3.1, 'JT': 4.2, 'CA': 0, 'CP': 3.5, 'CN': 2.7, 'BS': 6.5, 'SC': 2.6, 'PC': 3.8, 'TM': 1.3, 'AC': 4.9},
            'CP': {'TE': 5.0, 'ML': 1.3, 'AT': 4.8, 'MO': 2.5, 'JT': 2.0, 'CA': 3.5, 'CP': 0, 'CN': 0.85, 'BS': 3.7, 'SC': 0.9, 'PC': 2.7, 'TM': 3.4, 'AC': 3.8},
            'CN': {'TE': 4.4, 'ML': 1.1, 'AT': 4.9, 'MO': 2.0, 'JT': 2.4, 'CA': 2.7, 'CP': 0.85, 'CN': 0, 'BS': 4.5, 'SC': 0.4, 'PC': 2.8, 'TM': 2.7, 'AC': 3.9},
            'BS': {'TE': 5.5, 'ML': 3.4, 'AT': 4.3, 'MO': 3.9, 'JT': 2.7, 'CA': 6.5, 'CP': 3.7, 'CN': 4.5, 'BS': 0, 'SC': 4.2, 'PC': 3.3, 'TM': 5.7, 'AC': 3.8},
            'SC': {'TE': 4.2, 'ML': 0.8, 'AT': 4.6, 'MO': 1.8, 'JT': 2.0, 'CA': 2.6, 'CP': 0.9, 'CN': 0.4, 'BS': 4.2, 'SC': 0, 'PC': 2.5, 'TM': 2.6, 'AC': 3.6},
            'PC': {'TE': 2.5, 'ML': 1.7, 'AT': 2.2, 'MO': 1.0, 'JT': 1.0, 'CA': 3.8, 'CP': 2.7, 'CN': 2.8, 'BS': 3.3, 'SC': 2.5, 'PC': 0, 'TM': 3.0, 'AC': 1.2},
            'TM': {'TE': 3.1, 'ML': 2.5, 'AT': 4.4, 'MO': 2.3, 'JT': 3.4, 'CA': 1.3, 'CP': 3.4, 'CN': 2.7, 'BS': 5.7, 'SC': 2.6, 'PC': 3.0, 'TM': 0, 'AC': 2.1},
            'AC': {'TE': 1.9, 'ML': 2.8, 'AT': 1.0, 'MO': 2.1, 'JT': 2.1, 'CA': 4.9, 'CP': 3.8, 'CN': 3.9, 'BS': 3.8, 'SC': 3.6, 'PC': 1.2, 'TM': 2.1, 'AC': 0}
        }

class ParisTouristApp:
    def __init__(self, root):
        self.root = root
        self.root.title("How to visit Paris? - Lucia FERNANDEZ")
        self.root.geometry("1000x700")
        self.root.configure(bg='lightgray')
        
        self.optimizer = ParisTouristOptimizer()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main notebook for different sections
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Question 1 Tab
        q1_frame = ttk.Frame(notebook)
        notebook.add(q1_frame, text="Question 1 - Basic Optimization")
        self.create_question1_tab(q1_frame)
        
        # Question 2 Tab
        q2_frame = ttk.Frame(notebook)
        notebook.add(q2_frame, text="Question 2 - Preferences")
        self.create_question2_tab(q2_frame)
        
        # Question 3 Tab
        q3_frame = ttk.Frame(notebook)
        notebook.add(q3_frame, text="Question 3 - Rankings")
        self.create_question3_tab(q3_frame)
        
        # Results area
        self.results_text = scrolledtext.ScrolledText(self.root, height=15, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Display initial info in results
        self.display_site_info()
    
    def create_question1_tab(self, parent):
        # Budget and duration inputs
        input_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Maximum Budget (€):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.budget_var = tk.DoubleVar(value=75.0)
        ttk.Entry(input_frame, textvariable=self.budget_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Maximum Duration (h):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.duration_var = tk.DoubleVar(value=14.0)
        ttk.Entry(input_frame, textvariable=self.duration_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Solve Basic Optimization", 
                  command=self.solve_basic_optimization).grid(row=0, column=4, padx=10, pady=5)
        
        # Specific scenarios
        scenarios_frame = ttk.LabelFrame(parent, text="Predefined Scenarios", padding="10")
        scenarios_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scenario buttons frame
        scenario_buttons_frame = ttk.Frame(scenarios_frame)
        scenario_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(scenario_buttons_frame, text="1(a) - ListVisit 1", 
                  command=lambda: self.solve_scenario(75, 14, "1(a)")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(scenario_buttons_frame, text="1(b) - ListVisit 2", 
                  command=lambda: self.solve_scenario(65, 14, "1(b)")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(scenario_buttons_frame, text="1(c) - ListVisit 3", 
                  command=lambda: self.solve_scenario(90, 10, "1(c)")).grid(row=0, column=2, padx=5, pady=5)
        
        # Sites display frame under scenarios
        sites_frame = ttk.LabelFrame(scenarios_frame, text="Available Tourist Sites", padding="10")
        sites_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a treeview to display sites in a table format
        columns = ('code', 'name', 'duration', 'price', 'appreciation')
        self.sites_tree = ttk.Treeview(sites_frame, columns=columns, show='headings', height=12)
        
        # Define headings
        self.sites_tree.heading('code', text='Code')
        self.sites_tree.heading('name', text='Site Name')
        self.sites_tree.heading('duration', text='Duration (h)')
        self.sites_tree.heading('price', text='Price (€)')
        self.sites_tree.heading('appreciation', text='Appreciation (★)') #here starts
        
        # Define columns
        self.sites_tree.column('code', width=60, anchor=tk.CENTER)
        self.sites_tree.column('name', width=200, anchor=tk.W)
        self.sites_tree.column('duration', width=80, anchor=tk.CENTER)
        self.sites_tree.column('price', width=80, anchor=tk.CENTER)
        self.sites_tree.column('appreciation', width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(sites_frame, orient=tk.VERTICAL, command=self.sites_tree.yview)
        self.sites_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Pack tree and scrollbar
        self.sites_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate the treeview with site data
        self.populate_sites_tree()
    
    def populate_sites_tree(self):
        """Populate the treeview with tourist site data"""
        # Clear existing items
        for item in self.sites_tree.get_children():
            self.sites_tree.delete(item)
        
        # Add sites to treeview
        for code, data in self.optimizer.sites.items():

           # stars = '★' * data['appreciation'] #here to review
            self.sites_tree.insert('', tk.END, values=(
                code,
                data['name'],
                data['duration'],
                f"{data['price']:.1f}",
                data['appreciation']
            ))
    
    def create_question2_tab(self, parent):
        preferences_frame = ttk.LabelFrame(parent, text="Individual Preferences", padding="10")
        preferences_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Individual preference buttons
        pref_buttons = [
            ("Preference 1 - Close Sites", self.solve_pref1),
            ("Preference 2 - TE & CA Required", self.solve_pref2),
            ("Preference 3 - AC → not SC", self.solve_pref3),
            ("Preference 4 - AT Required", self.solve_pref4),
            ("Preference 5 - ML → MO", self.solve_pref5)
        ]
        
        for i, (text, command) in enumerate(pref_buttons):
            ttk.Button(preferences_frame, text=text, command=command).grid(
                row=i//3, column=i%3, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Combined preferences
        combined_frame = ttk.LabelFrame(parent, text="Combined Preferences", padding="10")
        combined_frame.pack(fill=tk.X, padx=10, pady=10)
        
        combos = [
            ("Pref 1 + Pref 2", "2(b)", self.solve_pref_1_2),
            ("Pref 1 + Pref 3", "2(c)", self.solve_pref_1_3),
            ("Pref 1 + Pref 4", "2(d)", self.solve_pref_1_4),
            ("Pref 2 + Pref 5", "2(e)", self.solve_pref_2_5),
            ("Pref 3 + Pref 4", "2(f)", self.solve_pref_3_4),
            ("Pref 4 + Pref 5", "2(g)", self.solve_pref_4_5),
            ("Pref 1+2+4", "2(h)", self.solve_pref_1_2_4),
            ("Pref 2+3+5", "2(i)", self.solve_pref_2_3_5),
            ("Pref 2+3+4+5", "2(j)", self.solve_pref_2_3_4_5),
            ("Pref 1+2+4+5", "2(k)", self.solve_pref_1_2_4_5),
            ("All Preferences", "2(l)", self.solve_all_prefs)
        ]
        
        for i, (text, label, command) in enumerate(combos):
            ttk.Button(combined_frame, text=f"{label}: {text}", 
                      command=command).grid(row=i//3, column=i%3, padx=5, pady=2, sticky=tk.W+tk.E)
    
    def create_question3_tab(self, parent):
        analysis_frame = ttk.LabelFrame(parent, text="Ranking Analysis", padding="10")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(analysis_frame, text="Compare Rankings (Duration, Appreciation, Price)", 
                  command=self.analyze_rankings).pack(pady=10)
        
        ttk.Button(analysis_frame, text="Calculate Kendall Tau Correlation", 
                  command=self.calculate_kendall_tau).pack(pady=5)
        
        ttk.Button(analysis_frame, text="Calculate Spearman Correlation", 
                  command=self.calculate_spearman).pack(pady=5)
    
    def display_site_info(self):
        """Display basic information about all sites in the results area"""
        info_text = "Objective: Maximize number of sites visited within constraints.\n\n"
        self.results_text.insert(tk.END, info_text + "\n")
    
    def solve_basic_optimization(self):
        """Solve the basic optimization problem"""
        budget = self.budget_var.get()
        max_duration = self.duration_var.get()
        
        prob = pulp.LpProblem("Paris_Tourist_Optimization", pulp.LpMaximize)
        
        # Decision variables
        site_vars = {}
        for site in self.optimizer.sites.keys():
            site_vars[site] = pulp.LpVariable(site, cat='Binary')
        
        # Objective: maximize number of sites
        prob += pulp.lpSum([site_vars[site] for site in self.optimizer.sites.keys()])
        
        # Constraints
        # Budget constraint
        prob += pulp.lpSum([self.optimizer.sites[site]['price'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= budget
        
        # Duration constraint
        prob += pulp.lpSum([self.optimizer.sites[site]['duration'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= max_duration
        
        # Solve
        prob.solve()
        
        # Display results
        self.display_solution(prob, site_vars, f"Basic Optimization (Budget: €{budget}, Duration: {max_duration}h)")
    
    def solve_scenario(self, budget, duration, scenario_name):
        """Solve a specific scenario"""
        prob = pulp.LpProblem(f"Scenario_{scenario_name}", pulp.LpMaximize)
        
        site_vars = {}
        for site in self.optimizer.sites.keys():
            site_vars[site] = pulp.LpVariable(site, cat='Binary')
        
        # Objective: maximize number of sites
        prob += pulp.lpSum([site_vars[site] for site in self.optimizer.sites.keys()])
        
        # Constraints
        prob += pulp.lpSum([self.optimizer.sites[site]['price'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= budget
        
        prob += pulp.lpSum([self.optimizer.sites[site]['duration'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= duration
        
        prob.solve()
        
        self.display_solution(prob, site_vars, f"Scenario {scenario_name} (Budget: €{budget}, Duration: {duration}h)")
    
    def solve_with_preferences(self, preferences_description, additional_constraints):
        """Generic method to solve with given preferences"""
        budget = 75
        max_duration = 14
        
        prob = pulp.LpProblem("Paris_Tourist_With_Preferences", pulp.LpMaximize)
        
        site_vars = {}
        for site in self.optimizer.sites.keys():
            site_vars[site] = pulp.LpVariable(site, cat='Binary')
        
        # Objective: maximize number of sites
        prob += pulp.lpSum([site_vars[site] for site in self.optimizer.sites.keys()])
        
        # Basic constraints
        prob += pulp.lpSum([self.optimizer.sites[site]['price'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= budget
        
        prob += pulp.lpSum([self.optimizer.sites[site]['duration'] * site_vars[site] 
                           for site in self.optimizer.sites.keys()]) <= max_duration
        
        # Add additional constraints
        additional_constraints(prob, site_vars)
        
        prob.solve()
        
        self.display_solution(prob, site_vars, preferences_description)
    
    def solve_pref1(self):
        """Preference 1: Prefer pairs of sites within 1km radius"""
        def add_constraints(prob, site_vars):
            # Find all site pairs within 1km
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            # Add constraint to encourage visiting both sites in close pairs
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
        
        self.solve_with_preferences("Preference 1 - Close Sites (within 1km)", add_constraints)
    
    def solve_pref2(self):
        """Preference 2: Must visit TE and CA"""
        def add_constraints(prob, site_vars):
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
        
        self.solve_with_preferences("Preference 2 - Must visit TE and CA", add_constraints)
    
    def solve_pref3(self):
        """Preference 3: If AC then not SC"""
        def add_constraints(prob, site_vars):
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
        
        self.solve_with_preferences("Preference 3 - If AC then not SC", add_constraints)
    
    def solve_pref4(self):
        """Preference 4: Must visit AT"""
        def add_constraints(prob, site_vars):
            prob += site_vars['AT'] == 1, "Must_visit_AT"
        
        self.solve_with_preferences("Preference 4 - Must visit AT", add_constraints)
    
    def solve_pref5(self):
        """Preference 5: If ML then MO"""
        def add_constraints(prob, site_vars):
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        
        self.solve_with_preferences("Preference 5 - If ML then MO", add_constraints)
    
    def solve_pref_1_2(self):
        """Preferences 1 + 2"""
        def add_constraints(prob, site_vars):
            # Pref 1: Close sites
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            
            # Pref 2: Must visit TE and CA
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
        
        self.solve_with_preferences("Preferences 1 + 2", add_constraints)
    
    def solve_pref_1_3(self):
        """Preferences 1 + 3"""
        def add_constraints(prob, site_vars):
            # Pref 1
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            
            # Pref 3
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
        
        self.solve_with_preferences("Preferences 1 + 3", add_constraints)
    
    def solve_pref_1_4(self):
        def add_constraints(prob, site_vars):
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
        self.solve_with_preferences("Preferences 1 + 4", add_constraints)
    
    def solve_pref_2_5(self):
        def add_constraints(prob, site_vars):
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        self.solve_with_preferences("Preferences 2 + 5", add_constraints)
    
    def solve_pref_3_4(self):
        def add_constraints(prob, site_vars):
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
        self.solve_with_preferences("Preferences 3 + 4", add_constraints)
    
    def solve_pref_4_5(self):
        def add_constraints(prob, site_vars):
            prob += site_vars['AT'] == 1, "Must_visit_AT"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        self.solve_with_preferences("Preferences 4 + 5", add_constraints)
    
    def solve_pref_1_2_4(self):
        def add_constraints(prob, site_vars):
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
        self.solve_with_preferences("Preferences 1 + 2 + 4", add_constraints)
    
    def solve_pref_2_3_5(self):
        def add_constraints(prob, site_vars):
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        self.solve_with_preferences("Preferences 2 + 3 + 5", add_constraints)
    
    def solve_pref_2_3_4_5(self):
        def add_constraints(prob, site_vars):
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        self.solve_with_preferences("Preferences 2 + 3 + 4 + 5", add_constraints)
    
    def solve_pref_1_2_4_5(self):
        def add_constraints(prob, site_vars):
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        self.solve_with_preferences("Preferences 1 + 2 + 4 + 5", add_constraints)
    
    def solve_all_prefs(self):
        def add_constraints(prob, site_vars):
            # Pref 1
            close_pairs = []
            for site1 in self.optimizer.sites.keys():
                for site2 in self.optimizer.sites.keys():
                    if site1 != site2 and self.optimizer.distances[site1][site2] <= 1.0:
                        close_pairs.append((site1, site2))
            
            for site1, site2 in close_pairs:
                prob += site_vars[site1] + site_vars[site2] >= 1, f"Close_Pair_{site1}_{site2}"
            
            # Pref 2, 3, 4, 5
            prob += site_vars['TE'] == 1, "Must_visit_TE"
            prob += site_vars['CA'] == 1, "Must_visit_CA"
            prob += site_vars['AC'] + site_vars['SC'] <= 1, "AC_then_not_SC"
            prob += site_vars['AT'] == 1, "Must_visit_AT"
            prob += site_vars['ML'] <= site_vars['MO'], "ML_then_MO"
        
        self.solve_with_preferences("All Preferences (1+2+3+4+5)", add_constraints)
    
    def analyze_rankings(self):
        """Analyze and compare different rankings"""
        sites_list = list(self.optimizer.sites.keys())
        
        # Duration ranking (minimize - lower duration is better)
        duration_ranking = sorted(sites_list, 
                                 key=lambda x: self.optimizer.sites[x]['duration'])
        
        # Appreciation ranking (maximize - higher appreciation is better)
        appreciation_ranking = sorted(sites_list, 
                                     key=lambda x: -self.optimizer.sites[x]['appreciation'])
        
        # Price ranking (minimize - lower price is better)
        price_ranking = sorted(sites_list, 
                              key=lambda x: self.optimizer.sites[x]['price'])
        
        result = "Ranking Analysis:\n"
        result += "=" * 80 + "\n"
        
        result += "Duration Ranking (shorter to longer):\n"
        for i, site in enumerate(duration_ranking, 1):
            result += f"{i:2d}. {site}: {self.optimizer.sites[site]['duration']}h\n"
        
        result += "\nAppreciation Ranking (higher to lower):\n"
        for i, site in enumerate(appreciation_ranking, 1):
            result += f"{i:2d}. {site}: {self.optimizer.sites[site]['appreciation']}★\n"
        
        result += "\nPrice Ranking (cheaper to more expensive):\n"
        for i, site in enumerate(price_ranking, 1):
            result += f"{i:2d}. {site}: €{self.optimizer.sites[site]['price']:.2f}\n"
        
        self.results_text.insert(tk.END, result + "\n")
    
    def calculate_kendall_tau(self):
        """Calculate Kendall Tau correlation between rankings"""
        sites_list = list(self.optimizer.sites.keys())
        
        # Get rankings
        duration_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: self.optimizer.sites[x]['duration']))}
        
        appreciation_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: -self.optimizer.sites[x]['appreciation']))}
        
        price_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: self.optimizer.sites[x]['price']))}
        
        # Calculate Kendall Tau
        def kendall_tau(rank1, rank2):
            concordant = 0
            discordant = 0
            n = len(rank1)
            
            for i in range(n):
                for j in range(i+1, n):
                    site1, site2 = sites_list[i], sites_list[j]
                    if (rank1[site1] < rank1[site2] and rank2[site1] < rank2[site2]) or \
                       (rank1[site1] > rank1[site2] and rank2[site1] > rank2[site2]):
                        concordant += 1
                    else:
                        discordant += 1
            
            return (concordant - discordant) / (concordant + discordant)
        
        tau_dur_app = kendall_tau(duration_rank, appreciation_rank)
        tau_dur_price = kendall_tau(duration_rank, price_rank)
        tau_app_price = kendall_tau(appreciation_rank, price_rank)
        
        result = "Kendall Tau Rank Correlation:\n"
        result += "=" * 50 + "\n"
        result += f"Duration vs Appreciation: {tau_dur_app:.3f}\n"
        result += f"Duration vs Price: {tau_dur_price:.3f}\n"
        result += f"Appreciation vs Price: {tau_app_price:.3f}\n"
        
        self.results_text.insert(tk.END, result + "\n")
    
    def calculate_spearman(self):
        """Calculate Spearman correlation between rankings"""
        sites_list = list(self.optimizer.sites.keys())
        
        # Get rankings
        duration_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: self.optimizer.sites[x]['duration']))}
        
        appreciation_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: -self.optimizer.sites[x]['appreciation']))}
        
        price_rank = {site: i for i, site in enumerate(
            sorted(sites_list, key=lambda x: self.optimizer.sites[x]['price']))}
        
        def spearman_correlation(rank1, rank2):
            n = len(rank1)
            sum_d_sq = 0
            
            for site in sites_list:
                d = rank1[site] - rank2[site]
                sum_d_sq += d * d
            
            return 1 - (6 * sum_d_sq) / (n * (n * n - 1))
        
        rho_dur_app = spearman_correlation(duration_rank, appreciation_rank)
        rho_dur_price = spearman_correlation(duration_rank, price_rank)
        rho_app_price = spearman_correlation(appreciation_rank, price_rank)
        
        result = "Spearman Rank Correlation:\n"
        result += "=" * 50 + "\n"
        result += f"Duration vs Appreciation: {rho_dur_app:.3f}\n"
        result += f"Duration vs Price: {rho_dur_price:.3f}\n"
        result += f"Appreciation vs Price: {rho_app_price:.3f}\n"
        
        self.results_text.insert(tk.END, result + "\n")
    
    def display_solution(self, prob, site_vars, title):
        """Display the solution in a formatted way"""
        result = f"\n{title}\n"
        result += "=" * 80 + "\n"
        result += f"Status: {pulp.LpStatus[prob.status]}\n"
        
        if prob.status == pulp.LpStatusOptimal:
            selected_sites = []
            total_cost = 0
            total_duration = 0
            total_appreciation = 0
            
            for site in self.optimizer.sites.keys():
                if site_vars[site].varValue == 1:
                    selected_sites.append(site)
                    total_cost += self.optimizer.sites[site]['price']
                    total_duration += self.optimizer.sites[site]['duration']
                    total_appreciation += self.optimizer.sites[site]['appreciation']
            
            result += f"Number of sites selected: {len(selected_sites)}\n"
            result += f"Total cost: €{total_cost:.2f}\n"
            result += f"Total duration: {total_duration:.2f}h\n"
            result += f"Total appreciation: {total_appreciation}★\n\n"
            
            result += "Selected sites:\n"
            for site in selected_sites:
                site_data = self.optimizer.sites[site]
                result += f"  • {site}: {site_data['name']} | "
                result += f"Duration: {site_data['duration']}h | "
                result += f"Price: €{site_data['price']:.2f} | "
                result += f"Appreciation: {site_data['appreciation']}★\n"
        else:
            result += "No optimal solution found!\n"
        
        result += "\n"
        self.results_text.insert(tk.END, result)
        self.results_text.see(tk.END)

def main():
    root = tk.Tk()
    app = ParisTouristApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()