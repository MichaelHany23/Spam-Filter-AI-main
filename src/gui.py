import sys
import os
import tkinter as tk
from tkinter import messagebox, ttk
import joblib

# Resolve project files from this script's location instead of the current
# working directory. This keeps the GUI working when launched by double-click.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# Import the preprocess_text function from data_preprocessing
from src.data_preprocessing import preprocess_text

class SpamFilterAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Filter AI | Email Security")
        self.root.geometry("760x620")
        self.root.minsize(650, 520)
        self.root.configure(bg='#F4F7FB')
        self.root.bind('<Control-Return>', lambda event: self.process_email())
        self.root.bind('<Escape>', lambda event: self.delete_mail())

        self.colors = {
            'navy': '#16324F',
            'blue': '#2F80ED',
            'background': '#F4F7FB',
            'card': '#FFFFFF',
            'text': '#243447',
            'muted': '#66788A',
            'border': '#DCE5EF',
            'green': '#168A5B',
            'red': '#C43D50',
        }

        self._build_header()
        self._build_content()
        self._load_artifacts()

    def _build_header(self):
        """Create the title area and a short explanation of the app."""
        header = tk.Frame(self.root, bg=self.colors['navy'], padx=28, pady=20)
        header.pack(fill='x')

        tk.Label(
            header,
            text="Spam Filter AI",
            font=("Segoe UI", 25, "bold"),
            fg='white',
            bg=self.colors['navy'],
        ).pack(anchor='w')
        tk.Label(
            header,
            text="Check an email message for suspicious content",
            font=("Segoe UI", 11),
            fg='#D7E6F5',
            bg=self.colors['navy'],
        ).pack(anchor='w', pady=(3, 0))

    def _build_content(self):
        """Build the input card, action buttons, and result area."""
        content = tk.Frame(self.root, bg=self.colors['background'], padx=28, pady=24)
        content.pack(fill='both', expand=True)

        card = tk.Frame(
            content,
            bg=self.colors['card'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
        )
        card.pack(fill='both', expand=True)

        tk.Label(
            card,
            text="Email content",
            font=("Segoe UI", 13, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text'],
        ).pack(anchor='w', padx=22, pady=(20, 2))
        tk.Label(
            card,
            text="Paste the full message below for the most reliable result.",
            font=("Segoe UI", 10),
            bg=self.colors['card'],
            fg=self.colors['muted'],
        ).pack(anchor='w', padx=22)

        text_frame = tk.Frame(card, bg=self.colors['card'])
        text_frame.pack(fill='both', expand=True, padx=22, pady=(12, 8))
        self.email_text = tk.Text(
            text_frame,
            height=12,
            wrap='word',
            undo=True,
            font=("Segoe UI", 11),
            bg='#FBFCFE',
            fg=self.colors['text'],
            insertbackground=self.colors['blue'],
            relief='flat',
            padx=12,
            pady=10,
            highlightbackground=self.colors['border'],
            highlightthickness=1,
        )
        self.email_text.pack(side='left', fill='both', expand=True)
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.email_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.email_text.configure(yscrollcommand=scrollbar.set)
        self.email_text.bind('<KeyRelease>', self._update_character_count)

        self.character_count = tk.Label(
            card,
            text="0 characters",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['muted'],
        )
        self.character_count.pack(anchor='e', padx=24)

        button_frame = tk.Frame(card, bg=self.colors['card'])
        button_frame.pack(fill='x', padx=22, pady=(12, 8))
        self.submit_button = tk.Button(
            button_frame,
            text="  Analyze Email  ",
            command=self.process_email,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['blue'],
            fg='white',
            activebackground='#1769C2',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=8,
            pady=8,
        )
        self.submit_button.pack(side='left')
        self.delete_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.delete_mail,
            font=("Segoe UI", 11),
            bg='#EEF2F7',
            fg=self.colors['text'],
            activebackground='#DCE5EF',
            relief='flat',
            cursor='hand2',
            padx=18,
            pady=8,
        )
        self.delete_button.pack(side='left', padx=(10, 0))
        tk.Label(
            button_frame,
            text="Ctrl+Enter to analyze  |  Esc to clear",
            font=("Segoe UI", 9),
            bg=self.colors['card'],
            fg=self.colors['muted'],
        ).pack(side='right', pady=8)

        self.status_label = tk.Label(
            card,
            text="Ready to analyze an email.",
            font=("Segoe UI", 13, "bold"),
            bg='#EEF6FF',
            fg=self.colors['blue'],
            anchor='w',
            padx=16,
            pady=13,
        )
        self.status_label.pack(fill='x', padx=22, pady=(4, 22))

        tk.Label(
            self.root,
            text="Spam Filter AI  •  Machine-learning email screening",
            font=("Segoe UI", 9),
            bg=self.colors['background'],
            fg=self.colors['muted'],
            pady=8,
        ).pack()

    def _load_artifacts(self):
        """Load the trained files and show a useful error if they are missing."""
        try:
            model_path = os.path.join(PROJECT_ROOT, 'spam_detector_model.pkl')
            vectorizer_path = os.path.join(PROJECT_ROOT, 'tfidf_vectorizer.pkl')
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
        except (FileNotFoundError, OSError, ValueError, EOFError) as error:
            self.model = None
            self.vectorizer = None
            self.submit_button.config(state='disabled')
            self.status_label.config(
                text="Model files are missing. Run the training pipeline first.",
                bg='#FFF1F1',
                fg=self.colors['red'],
            )
            messagebox.showerror("Model unavailable", f"Could not load the trained model:\n{error}")

    def _update_character_count(self, event=None):
        count = len(self.email_text.get("1.0", "end-1c"))
        self.character_count.config(text=f"{count:,} character{'s' if count != 1 else ''}")

    def process_email(self):
        email_content = self.email_text.get("1.0", tk.END).strip()
        if not email_content:
            messagebox.showwarning("Empty Content", "Please paste an email into the text area before submitting.")
            return

        if self.model is None or self.vectorizer is None:
            self.status_label.config(
                text="The trained model is unavailable. Please retrain the model first.",
                bg='#FFF1F1',
                fg=self.colors['red'],
            )
            return

        try:
            preprocessed_content = preprocess_text(email_content)
            features = self.vectorizer.transform([preprocessed_content])
            prediction = int(self.model.predict(features)[0])
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction] * 100
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            self.status_label.config(
                text=f"Could not analyze this email: {error}",
                bg='#FFF1F1',
                fg=self.colors['red'],
            )
            return

        if prediction == 1:
            result_text = f"⚠  Likely SPAM  •  Confidence: {confidence:.1f}%"
            background, foreground = '#FFF1F1', self.colors['red']
        else:
            result_text = f"✓  Likely NOT SPAM  •  Confidence: {confidence:.1f}%"
            background, foreground = '#ECF9F3', self.colors['green']

        self.status_label.config(text=result_text, bg=background, fg=foreground)

    def delete_mail(self):
        self.email_text.delete("1.0", tk.END)
        self._update_character_count()
        self.status_label.config(
            text="Ready to analyze an email.",
            bg='#EEF6FF',
            fg=self.colors['blue'],
        )

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamFilterAIApp(root)
    root.mainloop()
