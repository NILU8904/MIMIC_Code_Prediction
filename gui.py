import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import joblib
from preprocess import preprocess_text
from features import FeatureExtractor
from train_model import load_model

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
import spacy
from PIL import Image, ImageTk
import time
import threading


class ICDCodePredictorGUI:
    def __init__(self, root, model_path, vectorizer_prefix, mlb_path):
        self.root = root
        self.root.title("ICD Code Predictor")
        self.root.geometry("1100x800")
        self.root.configure(bg='#f0f2f5')
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Header frame
        header_frame = tk.Frame(root, bg='#2c3e50', height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="ICD Code Predictor", font=('Helvetica', 20, 'bold'), 
                              fg='white', bg='#2c3e50')
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Loading animation setup
        self.loading = False
        self.loading_label = tk.Label(header_frame, text="", font=('Helvetica', 12), 
                                     fg='white', bg='#2c3e50')
        self.loading_label.pack(side=tk.RIGHT, padx=20, pady=20)
        
        # Main content frame
        main_frame = tk.Frame(root, bg='#f0f2f5')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Load spaCy model for biomedical/clinical entities if available, else fallback
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                messagebox.showerror("Error", "spaCy model not found. Please install en_core_web_sm.")
                return

        # Load model, feature extractor and MultiLabelBinarizer
        try:
            self.model = load_model(model_path)
            self.fe = FeatureExtractor()
            self.fe.load(vectorizer_prefix)
            self.mlb = joblib.load(mlb_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model files: {str(e)}")
            return

        # Input section
        input_frame = tk.LabelFrame(main_frame, text=" Clinical Note Input ", font=('Helvetica', 12, 'bold'),
                                   bg='#f0f2f5', fg='#2c3e50', relief=tk.GROOVE, bd=2)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.text_area = scrolledtext.ScrolledText(input_frame, width=90, height=12, wrap=tk.WORD, 
                                                  font=('Helvetica', 11), relief=tk.FLAT, bd=2,
                                                  highlightbackground='#bdc3c7', highlightcolor='#3498db', 
                                                  highlightthickness=1)
        self.text_area.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)
        
        # Button with modern style
        button_frame = tk.Frame(main_frame, bg='#f0f2f5')
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.predict_button = tk.Button(button_frame, text="Predict ICD Codes", command=self.predict_icd_codes, 
                                       font=('Helvetica', 12, 'bold'), bg='#3498db', fg='white',
                                       activebackground='#2980b9', activeforeground='white',
                                       relief=tk.FLAT, bd=0, padx=20, pady=10, cursor='hand2')
        self.predict_button.pack(pady=5)
        
        # Bind hover effects
        self.predict_button.bind("<Enter>", lambda e: self.on_enter(e, self.predict_button))
        self.predict_button.bind("<Leave>", lambda e: self.on_leave(e, self.predict_button))
        
        # Results section
        results_frame = tk.LabelFrame(main_frame, text=" Prediction Results ", font=('Helvetica', 12, 'bold'),
                                     bg='#f0f2f5', fg='#2c3e50', relief=tk.GROOVE, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Result text with scrollbar
        result_text_frame = tk.Frame(results_frame, bg='#f0f2f5')
        result_text_frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        
        tk.Label(result_text_frame, text="Predicted ICD Codes:", font=('Helvetica', 11, 'bold'),
                bg='#f0f2f5', fg='#2c3e50').pack(anchor=tk.W)
        
        self.result_text = tk.Text(result_text_frame, width=90, height=3, state='disabled', 
                                  font=('Helvetica', 11), relief=tk.FLAT, bd=2,
                                  highlightbackground='#bdc3c7', highlightthickness=1)
        self.result_text.pack(fill=tk.X, pady=(5, 0))
        
        # Confidence chart
        chart_frame = tk.Frame(results_frame, bg='#f0f2f5')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.fig.patch.set_facecolor('#f0f2f5')
        self.ax.set_facecolor('#f0f2f5')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial message on chart
        self.ax.text(0.5, 0.5, "Enter clinical text and click 'Predict' to see results", 
                    ha='center', va='center', fontsize=14, alpha=0.5, transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()
        
        # Highlighting tags
        self.text_area.tag_configure("highlight", background="#fffacd", foreground="black", 
                                   font=('Helvetica', 11, 'bold'))
        self.text_area.tag_configure("fade", background="#f0f2f5")
        
        # Status bar
        self.status_bar = tk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W, 
                                  font=('Helvetica', 9), bg='#ecf0f1', fg='#7f8c8d')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Tooltip for button
        self.tooltip = None
        
    def configure_styles(self):
        # Configure ttk styles
        self.style.configure('TFrame', background='#f0f2f5')
        self.style.configure('TLabel', background='#f0f2f5', foreground='#2c3e50')
        self.style.configure('TButton', font=('Helvetica', 11), background='#3498db')
        self.style.map('TButton', background=[('active', '#2980b9')])
        
    def on_enter(self, event, button):
        button.configure(bg='#2980b9')
        # Show tooltip
        if button == self.predict_button:
            self.show_tooltip("Click to analyze the clinical text and predict ICD codes")
        
    def on_leave(self, event, button):
        button.configure(bg='#3498db')
        # Hide tooltip
        self.hide_tooltip()
        
    def show_tooltip(self, text):
        if self.tooltip:
            self.hide_tooltip()
            
        x, y, _, _ = self.predict_button.bbox("insert")
        x += self.predict_button.winfo_rootx() + 25
        y += self.predict_button.winfo_rooty() + 25
        
        self.tooltip = tk.Toplevel(self.predict_button)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip, text=text, background="#ffffe0", relief=tk.SOLID, 
                        borderwidth=1, font=('Helvetica', 10))
        label.pack()
        
    def hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None
            
    def animate_loading(self):
        dots = ""
        while self.loading:
            dots = dots + "." if len(dots) < 3 else ""
            self.loading_label.config(text=f"Processing{dots}")
            time.sleep(0.5)
            self.root.update()
        self.loading_label.config(text="")
        
    def highlight_medical_entities(self, text):
        """
        Use spaCy NER to highlight diseases/conditions/symptoms in input clinical note.
        """
        # First remove all existing tags
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.text_area.tag_remove("fade", "1.0", tk.END)
        
        # Apply fade effect to all text first
        self.text_area.tag_add("fade", "1.0", tk.END)
        
        doc = self.nlp(text)

        # Typical clinical entity labels; adjust based on the NER model used
        medical_labels = {"DISEASE", "CONDITION", "SYMPTOM", "MEDICALCONDITION", "PROBLEM"}

        for ent in doc.ents:
            # Highlight if entity label matches clinical terms
            label_valid = (ent.label_ in medical_labels) or (ent.label_.lower() in medical_labels)
            if label_valid:
                start_index = f"1.0 + {ent.start_char}c"
                end_index = f"1.0 + {ent.end_char}c"
                
                # Animate the highlighting
                for i in range(3):
                    color = "#fffacd" if i % 2 == 0 else "#ffeb3b"
                    self.text_area.tag_configure("highlight", background=color)
                    self.text_area.tag_add("highlight", start_index, end_index)
                    self.root.update()
                    time.sleep(0.1)
                
    def predict_icd_codes(self):
        # Start loading animation in a separate thread
        self.loading = True
        loading_thread = threading.Thread(target=self.animate_loading)
        loading_thread.daemon = True
        loading_thread.start()
        
        # Disable button during processing
        self.predict_button.config(state=tk.DISABLED, bg='#bdc3c7')
        self.status_bar.config(text="Processing...")
        
        # Process in separate thread to prevent GUI freezing
        def process_prediction():
            try:
                input_text = self.text_area.get("1.0", tk.END).strip()
                if not input_text:
                    messagebox.showwarning("Input Error", "Please enter clinical note text for prediction.")
                    return

                clean_text = preprocess_text(input_text)
                X = self.fe.transform([clean_text])
                y_prob = self.model.predict_proba(X)

                threshold = 0.5
                y_pred_labels = (y_prob >= threshold).astype(int)
                predicted_icd_codes = self.mlb.inverse_transform(y_pred_labels)

                # Update GUI in main thread
                self.root.after(0, self.update_results, input_text, y_prob, predicted_icd_codes, threshold)
                
            except Exception as e:
                self.root.after(0, self.show_error, str(e))
            finally:
                self.loading = False
                self.root.after(0, self.enable_button)
                
        # Start processing thread
        process_thread = threading.Thread(target=process_prediction)
        process_thread.daemon = True
        process_thread.start()
        
    def enable_button(self):
        self.predict_button.config(state=tk.NORMAL, bg='#3498db')
        self.status_bar.config(text="Ready")
        
    def show_error(self, error_msg):
        messagebox.showerror("Error", f"An error occurred: {error_msg}")
        self.status_bar.config(text="Error occurred")
        
    def update_results(self, input_text, y_prob, predicted_icd_codes, threshold):
        # Display prediction text
        self.result_text.config(state='normal')
        self.result_text.delete("1.0", tk.END)

        if predicted_icd_codes and predicted_icd_codes[0]:
            result_str = ", ".join(predicted_icd_codes[0])
            self.result_text.insert(tk.END, result_str)
        else:
            self.result_text.insert(tk.END, "No ICD codes predicted with confidence above threshold.")

        self.result_text.config(state='disabled')

        # Plot Bar Chart with color gradient and labels
        self.ax.clear()

        codes = self.mlb.classes_
        probs = y_prob[0]

        code_prob_pairs = [(c, p) for c, p in zip(codes, probs) if p >= threshold]
        if not code_prob_pairs:
            self.ax.text(0.5, 0.5, "No ICD codes above threshold for visualization.", 
                        ha='center', fontsize=12, color='#7f8c8d')
            self.ax.axis('off')
            self.canvas.draw()
            # Highlight without predictions
            self.highlight_medical_entities(input_text)
            return

        # Sort descending by confidence score
        code_prob_pairs.sort(key=lambda x: x[1], reverse=True)
        # Limit to top 10 for better visualization
        code_prob_pairs = code_prob_pairs[:10]
        codes_filtered, probs_filtered = zip(*code_prob_pairs)

        y_pos = range(len(codes_filtered))

        # Use a color gradient
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=min(probs_filtered), vmax=max(probs_filtered))
        colors = [cmap(norm(p)) for p in probs_filtered]

        bars = self.ax.barh(y_pos, probs_filtered, color=colors, edgecolor='white', alpha=0.8, height=0.6)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(codes_filtered, fontsize=11)
        self.ax.invert_yaxis()  # Highest scores on top

        self.ax.set_xlim(0, 1)
        self.ax.set_xlabel("Confidence Score", fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel("ICD Codes", fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_title("Predicted ICD Codes Confidence", fontsize=14, fontweight='bold', color='#2c3e50')

        # Style the chart
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#bdc3c7')
        self.ax.spines['bottom'].set_color('#bdc3c7')
        self.ax.tick_params(colors='#2c3e50')
        
        self.ax.grid(axis='x', linestyle='--', alpha=0.3, color='#bdc3c7')

        # Add score labels on bars
        for bar, score in zip(bars, probs_filtered):
            width = bar.get_width()
            self.ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{score:.2f}", 
                        va='center', fontsize=10, color='#2c3e50')

        self.fig.tight_layout()
        self.canvas.draw()

        # Highlight medical terms in the input text area with animation
        self.highlight_medical_entities(input_text)
        
        self.status_bar.config(text="Prediction complete")


def main():
    root = tk.Tk()
    model_path = "logreg_model.joblib"
    vectorizer_prefix = "feature_extractor"
    mlb_path = "mlb.joblib"
    
    # Set window icon if available
    try:
        img = Image.open('icon.png')  # Replace with your icon path if available
        photo = ImageTk.PhotoImage(img)
        root.wm_iconphoto(True, photo)
    except:
        pass
        
    app = ICDCodePredictorGUI(root, model_path, vectorizer_prefix, mlb_path)
    root.mainloop()


if __name__ == "__main__":
    main()