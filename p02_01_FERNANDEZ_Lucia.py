import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pulp

class VacationPackingOptimizer:
    def __init__(self):
        # Item data
        self.items = {
            'A': {'name': 'Item A', 'value': 10, 'weight': 5},
            'B': {'name': 'Item B', 'value': 8, 'weight': 7},
            'C': {'name': 'Item C', 'value': 12, 'weight': 4},
            'D': {'name': 'Item D', 'value': 4, 'weight': 3},
            'E': {'name': 'Item E', 'value': 5, 'weight': 5},
            'F': {'name': 'Item F', 'value': 10, 'weight': 3},
            'G': {'name': 'Item G', 'value': 6, 'weight': 4},
            'H': {'name': 'Item H', 'value': 9, 'weight': 6},
            'I': {'name': 'Item I', 'value': 7, 'weight': 4},
            'J': {'name': 'Item J', 'value': 9, 'weight': 6}
        }

class VacationPackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vacation Packing - Lucia FERNANDEZ")
        self.root.geometry("900x600")
        self.root.configure(bg='lightgray')
        
        self.optimizer = VacationPackingOptimizer()
        
        self.create_widgets()
        self.display_item_info()
    
    def create_widgets(self):
        # Main notebook for different sections
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic Optimization Tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Optimization")
        self.create_basic_tab(basic_frame)
        
        # Results area
        self.results_text = scrolledtext.ScrolledText(self.root, height=15, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_basic_tab(self, parent):
        # Weight input
        input_frame = ttk.LabelFrame(parent, text="Bag Parameters", padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Maximum Weight (kg):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.weight_var = tk.DoubleVar(value=23.0)
        ttk.Entry(input_frame, textvariable=self.weight_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Solve Optimization", 
                  command=self.solve_basic_optimization).grid(row=0, column=2, padx=10, pady=5)
    
    def display_item_info(self):
        """Display information about all items"""
        info_text = "Available Items:\n"
        info_text += "-" * 60 + "\n"
        for code, data in self.optimizer.items.items():
            info_text += f"{code}: {data['name']:8} | Value: €{data['value']:2} | "
            info_text += f"Weight: {data['weight']}kg\n"
        
        self.results_text.insert(tk.END, info_text + "\n")
    
    def solve_basic_optimization(self):
        """Solve the basic optimization problem"""
        max_weight = self.weight_var.get()
        
        prob = pulp.LpProblem("Vacation_Packing_Optimization", pulp.LpMaximize)
        
        # Decision variables
        item_vars = {}
        for item in self.optimizer.items.keys():
            item_vars[item] = pulp.LpVariable(item, cat='Binary')
        
        # Objective: maximize total value
        prob += pulp.lpSum([self.optimizer.items[item]['value'] * item_vars[item] 
                           for item in self.optimizer.items.keys()])
        
        # Weight constraint
        prob += pulp.lpSum([self.optimizer.items[item]['weight'] * item_vars[item] 
                           for item in self.optimizer.items.keys()]) <= max_weight
        
        # Solve
        prob.solve()
        
        # Display results
        self.display_solution(prob, item_vars, f"Basic Optimization (Max Weight: {max_weight}kg)")
    
    def display_solution(self, prob, item_vars, title):
        """Display the solution in a formatted way"""
        result = f"\n{title}\n"
        result += "=" * 60 + "\n"
        result += f"Status: {pulp.LpStatus[prob.status]}\n"
        
        if prob.status == pulp.LpStatusOptimal:
            selected_items = []
            total_value = 0
            total_weight = 0
            
            for item in self.optimizer.items.keys():
                if item_vars[item].varValue == 1:
                    selected_items.append(item)
                    total_value += self.optimizer.items[item]['value']
                    total_weight += self.optimizer.items[item]['weight']
            
            result += f"Number of items selected: {len(selected_items)}\n"
            result += f"Total value: €{total_value}\n"
            result += f"Total weight: {total_weight}kg\n\n"
            
            result += "Selected items:\n"
            for item in selected_items:
                item_data = self.optimizer.items[item]
                result += f"  • {item}: {item_data['name']} | "
                result += f"Value: €{item_data['value']} | "
                result += f"Weight: {item_data['weight']}kg\n"
        else:
            result += "No optimal solution found!\n"
        
        result += "\n"
        self.results_text.insert(tk.END, result)
        self.results_text.see(tk.END)

def main():
    root = tk.Tk()
    app = VacationPackingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()