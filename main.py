import os
import webbrowser
from audio_processor import AudioProcessor
from result_manager import ResultManager

def main():
    print("Аналіз аудіофайлів:")

    # Головний цикл програми
    while True:
        print("\n1 - Вказати свій аудіофайл")
        print("2 - Згенерувати тестовий файл і проаналізувати")
        print("0 - Вихід")
        choice = input("Ваш вибір: ").strip()

        if choice == "0":
            print("Роботу завершено.")
            break

        if choice not in ("1", "2"):
            print("Помилка: введіть 1, 2 або 0.")
            continue

        try:
            # Введення налаштувань
            try:
                sr_text = input("Частота дискретизації (Enter = 22050): ").strip()
                sr = int(sr_text) if sr_text != "" else 22050

                fft_text = input("Розмір вікна FFT (Enter = 2048): ").strip()
                n_fft = int(fft_text) if fft_text != "" else 2048
            except ValueError:
                raise ValueError("Потрібно ввести ціле число!")

            if sr <= 0 or n_fft <= 0:
                raise ValueError("Параметри повинні бути більшими за 0!")

            # Вибір: свій файл чи демо
            if choice == "1":
                path = input("Введіть шлях до файлу: ").strip().strip('"')
                proc = AudioProcessor(path, sr=sr, n_fft=n_fft)
            else:
                proc = AudioProcessor(sr=sr, n_fft=n_fft)
                demo_path = proc.create_demo_file("demo.wav")
                print("Створено тестовий файл:", demo_path)

            # Завантаження та аналіз
            print("Завантаження та очищення сигналу.")
            proc.load_and_clean()
            print(proc)

            print("Виконання розрахунків.")
            results = proc.analyze()

            print("\nОтримані результати:")
            for key, value in results.items():
                print(f"{key}: {value}")

            # Збереження графіків і звітів
            print("\nЗбереження графіків та звітів.")
            manager = ResultManager(proc, out_dir="output")
            plots = manager.save_plots()
            csv_file, json_file, html_file = manager.export_all(plots)

            print("Готово! Файли збережено у папку 'output':")
            print("-", csv_file)
            print("-", json_file)
            print("-", html_file)

            # Відкриття звіту в браузері
            abs_html_path = os.path.abspath(html_file)
            file_url = f"file:///{abs_html_path.replace(os.sep, '/')}"
            print(f"\nПосилання на звіт: {file_url}")

            while True:
                open_choice = input("Відкрити підсумковий звіт у браузері? (1 - Так / 2 - Ні): ").strip()
                if open_choice == "1":
                    webbrowser.open(file_url)
                    break
                elif open_choice == "2":
                    break
                else:
                    print("Помилка: введіть 1 або 2.")

        except FileNotFoundError as e:
            print("Помилка файлу:", e)
        except ValueError as e:
            print("Помилка даних:", e)
        except Exception as e:
            print("Сталася помилка:", e)


if __name__ == "__main__":

    main()
