import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


class ResultManager:
    # Клас для створення графіків та збереження звітів
    def __init__(self, processor, out_dir="output"):
        self._proc = processor
        self._out_dir = out_dir

        # Створення папки, якщо її немає
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

    def save_plots(self):
        signal = self._proc.get_signal()
        sr = self._proc.get_sr()

        # Графік №1 (Осцилограма)
        plt.figure(figsize=(9, 3))
        t = np.linspace(0, len(signal) / sr, len(signal))
        plt.plot(t, signal, color="blue", label="Амплітуда сигналу")
        plt.title("Осцилограма аудіосигналу")
        plt.xlabel("Час (с)")
        plt.ylabel("Амплітуда")
        plt.legend(loc="upper right")
        plt.grid(True)
        p1 = os.path.join(self._out_dir, "oscillogram.png")
        plt.tight_layout()
        plt.savefig(p1)
        plt.close()

        # Графік №2 (Спектрограма)
        plt.figure(figsize=(9, 3))
        img = librosa.display.specshow(self._proc.get_spec_db(), sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(img, format="%+2.0f дБ")
        plt.title("Спектрограма аудіосигналу")
        plt.xlabel("Час (с)")
        plt.ylabel("Частота (Гц)")
        plt.grid(True, alpha=0.3)
        p2 = os.path.join(self._out_dir, "spectrogram.png")
        plt.tight_layout()
        plt.savefig(p2)
        plt.close()

        # Графік №3 (Хроматограма)
        plt.figure(figsize=(9, 3))
        img = librosa.display.specshow(self._proc.get_chroma(), sr=sr, x_axis="time", y_axis="chroma")
        plt.colorbar(img)
        plt.title("Хроматограма (розподіл енергії по нотах)")
        plt.xlabel("Час (с)")
        plt.ylabel("Ноти")
        plt.grid(True, alpha=0.3)
        p3 = os.path.join(self._out_dir, "chromagram.png")
        plt.tight_layout()
        plt.savefig(p3)
        plt.close()

        return [p1, p2, p3]

    def generate_html_report(self, plot_paths):
        # Створення HTML-звіту
        results = self._proc.get_results()
        html_path = os.path.join(self._out_dir, "report.html")

        rows_html = ""
        for k, v in results.items():
            rows_html += f"<tr><td>{k}</td><td><b>{v}</b></td></tr>\n"

        p1_name = os.path.basename(plot_paths[0])
        p2_name = os.path.basename(plot_paths[1])
        p3_name = os.path.basename(plot_paths[2])

        html_content = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <title>Звіт аналізу аудіофайлу</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; color: #000000; }}
        h1, h2, h3 {{ color: #000000; }}
        table {{ border-collapse: collapse; width: 60%; margin-bottom: 25px; }}
        th, td {{ border: 1px solid #999999; padding: 8px 12px; text-align: left; color: #000000; }}
        th {{ background-color: #e6e6e6; }}
        img {{ max-width: 100%; border: 1px solid #cccccc; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Підсумковий звіт акустичного аналізу</h1>
    <h2>1. Розраховані характеристики</h2>
    <table>
        <tr><th>Параметр</th><th>Значення</th></tr>
        {rows_html}
    </table>
    <h2>2. Графічна візуалізація сигналу</h2>
    <h3>Осцилограма</h3>
    <img src="{p1_name}" alt="Осцилограма">
    <h3>Спектрограма</h3>
    <img src="{p2_name}" alt="Спектрограма">
    <h3>Хроматограма</h3>
    <img src="{p3_name}" alt="Хроматограма">
</body>
</html>"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def export_all(self, plot_paths):
        results = self._proc.get_results()

        # Збереження у CSV
        csv_path = os.path.join(self._out_dir, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Параметр", "Значення"])
            for k, v in results.items():
                writer.writerow([k, v])

        # Збереження у JSON
        json_path = os.path.join(self._out_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        # Створення HTML-звіту
        html_path = self.generate_html_report(plot_paths)

        return csv_path, json_path, html_path