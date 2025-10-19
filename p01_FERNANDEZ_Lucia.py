import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from collections import deque

class BinaryRelationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Preferences as Binary Relations by Lucia FERNANDEZ")
        self.root.geometry("900x800")
        self.root.configure(bg='lightgray')

        # Variables
        self.m = tk.IntVar(value=3)
        self.matrix_size = 3
        self.checkboxes = []
        self.relation_matrix = None

        # Dropdown variables
        self.basic_property_var = tk.StringVar(value="Choose Property")
        self.advanced_operation_var = tk.StringVar(value="Choose Operation")

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="5")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(input_frame, text="Number of elements:").grid(row=0, column=0, sticky=tk.W)
        size_spinbox = ttk.Spinbox(input_frame, from_=3, to=10, textvariable=self.m, width=8)
        size_spinbox.grid(row=0, column=1, padx=(5, 0))
        ttk.Button(input_frame, text="Apply", command=self.update_matrix_size).grid(row=0, column=2, padx=(5, 0))

        # Matrix section
        self.matrix_frame = ttk.LabelFrame(main_frame, text="Click to Define Relations", padding="10")
        self.matrix_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.create_matrix_grid()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10)

        ttk.Button(button_frame, text="Build Matrix", command=self.build_relation_matrix).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Show Matrix", command=self.show_matrix_window).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Visualize", command=self.visualize_relation).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Clear", command=self.clear_matrix).pack(side=tk.LEFT)

        # Property dropdowns
        self.create_property_dropdowns(main_frame)

        # Results area
        self.results_text = tk.Text(main_frame, height=10, width=80)
        self.results_text.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(10, 0))

        main_frame.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def create_matrix_grid(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.checkboxes = []
        size = self.matrix_size

        for i in range(size + 1):
            for j in range(size + 1):
                if i == 0 and j == 0:
                    ttk.Label(self.matrix_frame, text="", width=4).grid(row=0, column=0, padx=4, pady=4)
                elif i == 0:
                    ttk.Label(self.matrix_frame, text=chr(97 + j - 1),
                              font=('Arial', 10, 'bold'), anchor='center', width=4).grid(row=i, column=j, padx=4, pady=4)
                elif j == 0:
                    ttk.Label(self.matrix_frame, text=chr(97 + i - 1),
                              font=('Arial', 10, 'bold'), anchor='center', width=4).grid(row=i, column=j, padx=4, pady=4)
                else:
                    var = tk.BooleanVar()
                    if i - 1 == j - 1:
                        var.set(True)
                    cb = ttk.Checkbutton(self.matrix_frame, variable=var)
                    cb.grid(row=i, column=j, padx=6, pady=6)
                    if len(self.checkboxes) < i:
                        self.checkboxes.append([])
                    self.checkboxes[i - 1].append(var)

    def create_property_dropdowns(self, parent):
        properties_frame = ttk.LabelFrame(parent, text="Property Checking and Operations", padding="10")
        properties_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        basic_frame = ttk.Frame(properties_frame)
        basic_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(basic_frame, text="Basic Properties:", font=('Arial', 10, 'bold')).pack(anchor='w')

        basic_properties = [
            "Choose Property",
            "Check Completeness",
            "Check Reflexivity",
            "Check Symmetry",
            "Check Asymmetry",
            "Check Antisymmetry",
            "Check Transitivity",
            "Check Negative Transitivity"
        ]

        basic_combo = ttk.Combobox(basic_frame, textvariable=self.basic_property_var,
                                   values=basic_properties, state="readonly", width=30)
        basic_combo.pack(side=tk.LEFT, padx=(0, 10))
        basic_combo.bind('<<ComboboxSelected>>', self.on_basic_property_selected)
        ttk.Button(basic_frame, text="Check", command=self.execute_basic_property).pack(side=tk.LEFT)

        advanced_frame = ttk.Frame(properties_frame)
        advanced_frame.pack(fill=tk.X)
        ttk.Label(advanced_frame, text="Advanced Operations:", font=('Arial', 10, 'bold')).pack(anchor='w')

        advanced_operations = [
            "Choose Operation",
            "Check Complete Order",
            "Check Complete Pre-order",
            "Get StrictRelation",
            "Get IndifferenceRelation",
            "Get Topologicalsorting1",
            "Get Topologicalsorting2"
        ]

        advanced_combo = ttk.Combobox(advanced_frame, textvariable=self.advanced_operation_var,
                                      values=advanced_operations, state="readonly", width=30)
        advanced_combo.pack(side=tk.LEFT, padx=(0, 10))
        advanced_combo.bind('<<ComboboxSelected>>', self.on_advanced_operation_selected)
        ttk.Button(advanced_frame, text="Execute", command=self.execute_advanced_operation).pack(side=tk.LEFT)

    # --------------------------------------------------------------------------
    # Matrix operations
    # --------------------------------------------------------------------------
    def update_matrix_size(self):
        try:
            new_size = self.m.get()
            if 3 <= new_size <= 10:
                self.matrix_size = new_size
                self.create_matrix_grid()
            else:
                messagebox.showerror("Error", "Please enter a number between 3 and 10!")
        except:
            messagebox.showerror("Error", "Invalid input, try again!")

    def build_relation_matrix(self):
        size = self.matrix_size
        self.relation_matrix = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if self.checkboxes[i][j].get():
                    self.relation_matrix[i][j] = 1
        messagebox.showinfo("Success", "Relation matrix built successfully!")

    def show_matrix_window(self):
        if self.relation_matrix is None:
            messagebox.showerror("Error", "Please build the relation matrix first!")
            return

        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Current Relation Matrix")
        matrix_window.geometry("500x400")

        matrix_frame = ttk.Frame(matrix_window, padding="10")
        matrix_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(matrix_frame, text="Current Relation Matrix",
                  font=('Arial', 12, 'bold')).pack(pady=(0, 10))

        matrix_text = tk.Text(matrix_frame, height=15, width=50, font=('Courier', 10))
        matrix_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.display_matrix_in_text(matrix_text)
        ttk.Button(matrix_frame, text="Close", command=matrix_window.destroy).pack(pady=10)

    def display_matrix_in_text(self, text_widget):
        if self.relation_matrix is None:
            return
        text_widget.delete(1.0, tk.END)
        size = self.matrix_size
        header = "    " + "   ".join(chr(97 + i) for i in range(size)) + "\n"
        text_widget.insert(tk.END, header)
        text_widget.insert(tk.END, "   " + "---" * size + "\n")
        for i in range(size):
            row_str = f"{chr(97 + i)} | "
            for j in range(size):
                row_str += f" {self.relation_matrix[i][j]} "
            row_str += "\n"
            text_widget.insert(tk.END, row_str)

    def clear_matrix(self):
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                self.checkboxes[i][j].set(False)
        self.relation_matrix = None
        self.results_text.delete(1.0, tk.END)
        self.basic_property_var.set("Choose Property:_")
        self.advanced_operation_var.set("Choose Operation:_")

    def visualize_relation(self):
        if self.relation_matrix is None:
            messagebox.showerror("Error", "Please build the relation matrix first!")
            return

        viz_window = tk.Toplevel(self.root)
        viz_window.title("Relation Graph")
        viz_window.geometry("700x600")

        fig, ax = plt.subplots(figsize=(7, 6))
        G = nx.DiGraph()
        size = self.matrix_size
        nodes = [chr(97 + i) for i in range(size)]
        G.add_nodes_from(nodes)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] == 1:
                    G.add_edge(chr(97 + i), chr(97 + j))
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
        plt.title("Graph", fontsize=14)
        plt.axis('off')
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(viz_window, text="Close", command=viz_window.destroy).pack(pady=5)

    def on_basic_property_selected(self, event): self.execute_basic_property()
    def on_advanced_operation_selected(self, event): self.execute_advanced_operation()

    def execute_basic_property(self):
        property_name = self.basic_property_var.get()
        if property_name == "Choose Property": return
        mapping = {
            "Check Completeness": self.CompleteCheck,
            "Check Reflexivity": self.ReflexiveCheck,
            "Check Symmetry": self.SymmetricCheck,
            "Check Asymmetry": self.AsymmetricCheck,
            "Check Antisymmetry": self.AntisymmetricCheck,
            "Check Transitivity": self.TransitiveCheck,
            "Check Negative Transitivity": self.NegativetransitiveCheck
        }
        method = mapping.get(property_name)
        if method:
            result = method()
            self.show_property_result(result)

    def execute_advanced_operation(self):
        operation = self.advanced_operation_var.get()
        if operation == "Choose Operation": return
        mapping = {
            "Check Complete Order": lambda: self.show_property_result(self.CompleteOrderCheck()),
            "Check Complete Pre-order": lambda: self.show_property_result(self.CompletePreOrderCheck()),
            "Get StrictRelation": self.show_strict_relation,
            "Get IndifferenceRelation": self.show_indifference_relation,
            "Get Topologicalsorting1": self.show_topological_sort1,
            "Get Topologicalsorting2": self.show_topological_sort2
            
        }
        method = mapping.get(operation)
        if method:
            method()

    def show_property_result(self, result):
        if isinstance(result, tuple):
            success, message = result
            if success:
                self.results_text.insert(tk.END, f"✓ {message}\n")
            else:
                self.results_text.insert(tk.END, f"✗ {message}\n")
        self.results_text.see(tk.END)

    def CompleteCheck(self):
        """Complete: for every x,y ∈ X, we have xRy or yRx (or both)"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] == 0 and self.relation_matrix[j][i] == 0:
                    return False, f"Not complete: {chr(97+i)} and {chr(97+j)} are not related"
        return True, "Relation is complete"

    def ReflexiveCheck(self):
        """Reflexive: xRx for all x ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            if self.relation_matrix[i][i] == 0:
                return False, f"Not reflexive: {chr(97+i)} not related to itself"
        return True, "Relation is reflexive"

    def AsymmetricCheck(self):
        """Asymmetric: xRy ⇒ not(yRx) for all x,y ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] == 1 and self.relation_matrix[j][i] == 1:
                    return False, f"Not asymmetric: {chr(97+i)} and {chr(97+j)} both exist"
        return True, "Relation is asymmetric"

    def SymmetricCheck(self):
        """Symmetric: xRy ⇒ yRx for all x,y ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] != self.relation_matrix[j][i]:
                    return False, f"Not symmetric: {chr(97+i)}B{chr(97+j)} ≠ {chr(97+j)}B{chr(97+i)}"
        return True, "Relation is symmetric"

    def AntisymmetricCheck(self):
        """Antisymmetric: xRy ∧ yRx ⇒ x = y for all x,y ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            for j in range(size):
                if i != j and self.relation_matrix[i][j] == 1 and self.relation_matrix[j][i] == 1:
                    return False, f"Not antisymmetric: {chr(97+i)} and {chr(97+j)} both ways but {chr(97+i)} ≠ {chr(97+j)}"
        return True, "Relation is antisymmetric"

    def TransitiveCheck(self):
        """Transitive: xRy ∧ yRz ⇒ xRz for all x,y,z ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        # Using matrix multiplication: R is transitive if R² ⊆ R
        R_squared = np.dot(self.relation_matrix, self.relation_matrix)
        R_squared = (R_squared > 0).astype(int)
        for i in range(size):
            for j in range(size):
                if R_squared[i][j] == 1 and self.relation_matrix[i][j] == 0:
                    # Find the specific counterexample
                    for k in range(size):
                        if self.relation_matrix[i][k] == 1 and self.relation_matrix[k][j] == 1:
                            return False, f"Not transitive: {chr(97+i)}B{chr(97+k)} and {chr(97+k)}B{chr(97+j)} but not {chr(97+i)}B{chr(97+j)}"
        return True, "Relation is transitive"

    def NegativetransitiveCheck(self):
        """Negative Transitive: not(xRy) ∧ not(yRz) ⇒ not(xRz) for all x,y,z ∈ X"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        size = len(self.relation_matrix)
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # If not(iRj) and not(jRk) but iRk, then not negative transitive
                    if (self.relation_matrix[i][j] == 0 and 
                        self.relation_matrix[j][k] == 0 and 
                        self.relation_matrix[i][k] == 1):
                        return False, f"Not negative transitive: not({chr(97+i)}R{chr(97+j)}) and not({chr(97+j)}R{chr(97+k)}) but {chr(97+i)}R{chr(97+k)}"
        return True, "Relation is negative transitive"

    def CompleteOrderCheck(self):
        """Total Order: Complete + Antisymmetric + Transitive"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        complete, msg1 = self.CompleteCheck()
        if not complete: return False, "Not a total order: " + msg1
        antisym, msg2 = self.AntisymmetricCheck()
        if not antisym: return False, "Not a total order: " + msg2
        trans, msg3 = self.TransitiveCheck()
        if not trans: return False, "Not a total order: " + msg3
        return True, "Relation is a total order"

    def CompletePreOrderCheck(self):
        """Complete Pre-order: Complete + Transitive"""
        if self.relation_matrix is None:
            return False, "No matrix built"
        complete, msg1 = self.CompleteCheck()
        if not complete: return False, "Not a complete pre-order: " + msg1
        trans, msg2 = self.TransitiveCheck()
        if not trans: return False, "Not a complete pre-order: " + msg2
        return True, "Relation is a complete pre-order"

    def StrictRelation(self):
        """Asymmetric part: xPy iff xRy and not(yRx)"""
        if self.relation_matrix is None: return None
        size = len(self.relation_matrix)
        strict = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] == 1 and self.relation_matrix[j][i] == 0:
                    strict[i][j] = 1
        return strict

    def show_strict_relation(self):
        strict_matrix = self.StrictRelation()
        if strict_matrix is not None:
            self.results_text.insert(tk.END, "Asymmetric Part:\n")
            size = len(strict_matrix)
            for i in range(size):
                for j in range(size):
                    if strict_matrix[i][j] == 1:
                        self.results_text.insert(tk.END, f"  {chr(97+i)}P{chr(97+j)}\n")
            self.results_text.see(tk.END)

    def IndifferenceRelation(self):
        """Symmetric part: xIy iff xRy and yRx"""
        if self.relation_matrix is None: return None
        size = len(self.relation_matrix)
        indifference = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if self.relation_matrix[i][j] == 1 and self.relation_matrix[j][i] == 1:
                    indifference[i][j] = 1
        return indifference

    def show_indifference_relation(self):
        indifference_matrix = self.IndifferenceRelation()
        if indifference_matrix is not None:
            self.results_text.insert(tk.END, "Symmetric Part:\n")
            size = len(indifference_matrix)
            for i in range(size):
                for j in range(i, size):
                    if indifference_matrix[i][j] == 1:
                        self.results_text.insert(tk.END, f"  {chr(97+i)}I{chr(97+j)}\n")
            self.results_text.see(tk.END)

    def Topologicalsorting1(self):
        """
        Return a topological ordering (list of labels) of the strict relation
        (asymmetric part). If there's a cycle (including self-cycles), return None.
        """
        if self.relation_matrix is None:
            return None

        strict = self.StrictRelation()
        if strict is None:
            return None

        n = len(strict)
        
 
        for i in range(n):
            if strict[i][i] == 1:
                return None  
        
  
        if self.has_cycle_in_strict(strict):
            return None  

        in_degree = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and strict[u][v] == 1:  # Skip self-relations
                    in_degree[v] += 1

        # Kahn's algorithm
        q = deque([i for i in range(n) if in_degree[i] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(chr(97 + u))
            for v in range(n):
                if u != v and strict[u][v] == 1:  # Skip self-relations
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        q.append(v)

        if len(topo) != n:
            return None  # cycle detected
        return topo

    def has_cycle_in_strict(self, strict_matrix):
        """
        Check if there's a cycle in the strict relation using DFS
        Returns True if cycle exists, False otherwise
        """
        n = len(strict_matrix)
        visited = [0] * n 
        
        def dfs(node):
            if visited[node] == 1:  
                return True
            if visited[node] == 2:  
                return False
                
            visited[node] = 1  
            
            for neighbor in range(n):
                if node != neighbor and strict_matrix[node][neighbor] == 1:
                    if dfs(neighbor):
                        return True
                        
            visited[node] = 2  
            return False
        
 
        for node in range(n):
            if visited[node] == 0:
                if dfs(node):
                    return True
        return False

 
    def Topologicalsorting2(self):
        """
        Build equivalence classes from the indifference relation (mutual relations)
        and topologically sort those classes according to strict edges between classes.
        Returns a list of classes (each class is a list of labels), or None if cycle.
        """
        if self.relation_matrix is None:
            return None

        size = len(self.relation_matrix)
        indiff = self.IndifferenceRelation()
        strict = self.StrictRelation()
        if indiff is None or strict is None:
            return None

       
        for i in range(size):
            if strict[i][i] == 1:
                return None  

       
        visited = [False] * size
        classes = []
        for i in range(size):
            if not visited[i]:
               
                stack = [i]
                comp = []
                visited[i] = True
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in range(size):
                        if indiff[u][v] == 1 and not visited[v]:
                            visited[v] = True
                            stack.append(v)
                classes.append(sorted(comp))


        elem_to_class = {}
        for ci, comp in enumerate(classes):
            for e in comp:
                elem_to_class[e] = ci

        
        class_count = len(classes)
        class_graph = [[0] * class_count for _ in range(class_count)]
        for u in range(size):
            for v in range(size):
                if u != v and strict[u][v] == 1:
                    cu = elem_to_class[u]
                    cv = elem_to_class[v]
                    if cu != cv:
                        class_graph[cu][cv] = 1

        
        if self.has_cycle_in_class_graph(class_graph, class_count):
            return None


        in_degree = [0] * class_count
        for i in range(class_count):
            for j in range(class_count):
                if class_graph[i][j] == 1:
                    in_degree[j] += 1

        q = deque([i for i in range(class_count) if in_degree[i] == 0])
        class_order = []
        while q:
            ci = q.popleft()
      
            class_order.append([chr(97 + e) for e in classes[ci]])
            for j in range(class_count):
                if class_graph[ci][j] == 1:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        q.append(j)

        if len(class_order) != class_count:
            return None  
        return class_order

    def has_cycle_in_class_graph(self, class_graph, class_count):
        """
        Check if there's a cycle in the class-level graph using DFS
        """
        visited = [0] * class_count 
        
        def dfs(node):
            if visited[node] == 1: 
                return True
            if visited[node] == 2:  
                return False
                
            visited[node] = 1  
            
            for neighbor in range(class_count):
                if class_graph[node][neighbor] == 1:
                    if dfs(neighbor):
                        return True
                        
            visited[node] = 2 
            return False
        
        
        for node in range(class_count):
            if visited[node] == 0:
                if dfs(node):
                    return True
        return False
    
    def show_topological_sort1(self):
        order = self.Topologicalsorting1()
        if order:
            self.results_text.insert(tk.END, f"Topological Sort (Strict): {' → '.join(order)}\n")
        else:
            self.results_text.insert(tk.END, "No topological sort (strict): cycle detected or matrix missing\n")
        self.results_text.see(tk.END)

    def show_topological_sort2(self):
        order = self.Topologicalsorting2()
        if order:
            formatted = " → ".join(["{" + ",".join(group) + "}" for group in order])
            self.results_text.insert(tk.END, f"Topological Sort (With Indifference): {formatted}\n")
        else:
            self.results_text.insert(tk.END, "No topological sort (with indifference): cycle detected or matrix missing\n")
        self.results_text.see(tk.END)


def main():
    try:
        root = tk.Tk()
        app = BinaryRelationApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()