import streamlit as st
import pandas as pd
import time
import os
from transformers import pipeline

# Konfiguracja strony
st.set_page_config(
    page_title="NLP Lab - Translator & Sentiment Analysis",
    page_icon="🌐",
    layout="wide"
)

# Nagłówek aplikacji
st.title('🌐 NLP Lab - Aplikacja do przetwarzania języka naturalnego')
st.markdown('---')

# Instrukcja
st.header('📖 Instrukcja użytkowania')
st.info("""
**Witaj w aplikacji NLP Lab!**

Ta aplikacja oferuje dwie główne funkcjonalności:
1. **Analiza wydźwięku emocjonalnego** - Sprawdź czy tekst w języku angielskim ma pozytywny czy negatywny wydźwięk
2. **Tłumaczenie EN→DE** - Przetłumacz tekst z języka angielskiego na język niemiecki

**Jak korzystać:**
- Wybierz opcję z menu rozwijanego
- Wpisz tekst w pole tekstowe
- Poczekaj na wynik analizy lub tłumaczenia
""")

st.markdown('---')

# Sukces uruchomienia
st.success('✅ Aplikacja uruchomiona pomyślnie!')

st.header('🔧 Przetwarzanie języka naturalnego')

# Menu wyboru opcji
option = st.selectbox(
    "Wybierz funkcjonalność:",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie angielski → niemiecki",
    ],
)

# Opcja 1: Analiza sentymentu
if option == "Wydźwięk emocjonalny tekstu (eng)":
    st.subheader('😊 Analiza wydźwięku emocjonalnego')
    st.write('Wpisz tekst w języku angielskim, aby sprawdzić jego wydźwięk emocjonalny.')

    text = st.text_area(label="Twój tekst (po angielsku):", height=150)

    if text:
        try:
            with st.spinner('🔄 Analizuję tekst...'):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)

            st.success('✅ Analiza zakończona!')

            # Wyświetlenie wyników w czytelny sposób
            result = answer[0]
            sentiment = result['label']
            confidence = result['score']

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wydźwięk", sentiment)
            with col2:
                st.metric("Pewność", f"{confidence:.2%}")

            # Kolorowe tło w zależności od wyniku
            if sentiment == "POSITIVE":
                st.balloons()

        except Exception as e:
            st.error(f'❌ Wystąpił błąd podczas analizy: {str(e)}')

# Opcja 2: Tłumaczenie EN→DE
elif option == "Tłumaczenie angielski → niemiecki":
    st.subheader('🇬🇧 → 🇩🇪 Tłumaczenie tekstu')
    st.write('Wpisz tekst w języku angielskim, a zostanie przetłumaczony na niemiecki.')

    text = st.text_area(label="Twój tekst (po angielsku):", height=150)

    if text:
        try:
            with st.spinner('🔄 Tłumaczę tekst...'):
                # Używamy modelu Helsinki-NLP dla tłumaczenia EN→DE
                translator = pipeline("translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de")
                translation = translator(text)

            st.success('✅ Tłumaczenie zakończone!')

            # Wyświetlenie tłumaczenia
            st.subheader('📝 Przetłumaczony tekst:')
            st.write(translation[0]['translation_text'])

            # Informacja dodatkowa
            st.info('💡 Tłumaczenie wykonane przy użyciu modelu Helsinki-NLP/opus-mt-en-de')

        except Exception as e:
            st.error(f'❌ Wystąpił błąd podczas tłumaczenia: {str(e)}')
            st.warning('⚠️ Upewnij się, że masz zainstalowane wszystkie wymagane biblioteki.')

st.markdown('---')

# Sekcja z danymi (oryginalny kod)
st.header('📊 Przykładowe dane')
st.write('Poniżej prezentujemy przykładowy zbiór danych:')

try:
    df = pd.read_csv("DSP_4.csv", sep=';')
    st.dataframe(df)
except FileNotFoundError:
    st.warning('⚠️ Plik DSP_4.csv nie został znaleziony.')
except Exception as e:
    st.error(f'❌ Błąd podczas wczytywania danych: {str(e)}')

st.markdown('---')

# Numer indeksu
st.header('👨‍🎓 Informacje o autorze')
st.write('**Numer indeksu:** s27600')
st.write('**Projekt:** Lab05 - Streamlit + Hugging Face Transformers')
st.write('**Data:** 2026-05-15')
