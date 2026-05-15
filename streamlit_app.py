import streamlit as st
import pandas as pd
import time
import os
from transformers import pipeline, MarianMTModel, MarianTokenizer

# Konfiguracja strony
st.set_page_config(
    page_title="NLP Lab - Translator & Sentiment Analysis",
    page_icon="🌐",
    layout="wide"
)

# Inicjalizacja session state dla historii
if 'history' not in st.session_state:
    st.session_state.history = []

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
- Użyj przykładowych tekstów lub wpisz własny
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

    # Przykładowe teksty
    st.markdown("**Przykładowe teksty do wypróbowania:**")
    col1, col2, col3 = st.columns(3)

    # Inicjalizacja default text
    if 'sentiment_text' not in st.session_state:
        st.session_state.sentiment_text = ''

    with col1:
        if st.button("😊 Pozytywny", use_container_width=True):
            st.session_state.sentiment_text = "I absolutely love this application! It's amazing and works perfectly."
    with col2:
        if st.button("😢 Negatywny", use_container_width=True):
            st.session_state.sentiment_text = "This is terrible and I'm very disappointed with the results."
    with col3:
        if st.button("😐 Neutralny", use_container_width=True):
            st.session_state.sentiment_text = "The weather is cloudy today."

    # Pole tekstowe
    text = st.text_area(
        label="Twój tekst (po angielsku):",
        height=150,
        value=st.session_state.sentiment_text
    )

    # Synchronizuj z session_state
    if text != st.session_state.sentiment_text:
        st.session_state.sentiment_text = text

    # Statystyki tekstu
    if text:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Liczba znaków", len(text))
        with col2:
            st.metric("🔤 Liczba słów", len(text.split()))
        with col3:
            st.metric("📏 Liczba zdań", text.count('.') + text.count('!') + text.count('?'))

    col_analyze, col_clear = st.columns([3, 1])
    with col_analyze:
        analyze_button = st.button("🔍 Analizuj", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🗑️ Wyczyść", use_container_width=True):
            st.session_state.sentiment_text = ""
            st.rerun()

    if text and analyze_button:
        try:
            with st.spinner('🔄 Analizuję tekst...'):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)

            st.success('✅ Analiza zakończona!')

            # Wyświetlenie wyników w czytelny sposób
            result = answer[0]
            sentiment = result['label']
            confidence = result['score']

            # Kolorowe wyświetlenie wyniku
            if sentiment == "POSITIVE":
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;">
                    <h1 style="margin: 0; font-size: 3em;">😊 {sentiment}</h1>
                    <h2 style="margin: 10px 0 0 0;">Pewność: {confidence:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 30px; border-radius: 15px; text-align: center; color: white; margin: 20px 0;">
                    <h1 style="margin: 0; font-size: 3em;">😢 {sentiment}</h1>
                    <h2 style="margin: 10px 0 0 0;">Pewność: {confidence:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)

            # Dodaj do historii
            st.session_state.history.append({
                'type': 'Sentiment Analysis',
                'input': text[:50] + '...' if len(text) > 50 else text,
                'output': f"{sentiment} ({confidence:.2%})",
                'time': time.strftime("%H:%M:%S")
            })

        except Exception as e:
            st.error(f'❌ Wystąpił błąd podczas analizy: {str(e)}')

# Opcja 2: Tłumaczenie EN→DE
elif option == "Tłumaczenie angielski → niemiecki":
    st.subheader('🇬🇧 → 🇩🇪 Tłumaczenie tekstu')
    st.write('Wpisz tekst w języku angielskim, a zostanie przetłumaczony na niemiecki.')

    # Przykładowe teksty
    st.markdown("**Przykładowe teksty do wypróbowania:**")
    col1, col2, col3 = st.columns(3)

    # Inicjalizacja default text
    if 'translation_text' not in st.session_state:
        st.session_state.translation_text = ''

    with col1:
        if st.button("👋 Powitanie", use_container_width=True):
            st.session_state.translation_text = "Hello! How are you today? I hope you're doing well."
    with col2:
        if st.button("🍕 Restauracja", use_container_width=True):
            st.session_state.translation_text = "I would like to order a pizza with extra cheese, please."
    with col3:
        if st.button("🌍 Podróż", use_container_width=True):
            st.session_state.translation_text = "Where is the nearest train station? I need to buy a ticket."

    # Pole tekstowe
    text = st.text_area(
        label="Twój tekst (po angielsku):",
        height=150,
        value=st.session_state.translation_text
    )

    # Synchronizuj z session_state
    if text != st.session_state.translation_text:
        st.session_state.translation_text = text

    # Statystyki tekstu
    if text:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Liczba znaków", len(text))
        with col2:
            st.metric("🔤 Liczba słów", len(text.split()))
        with col3:
            st.metric("📏 Liczba zdań", text.count('.') + text.count('!') + text.count('?'))

    col_translate, col_clear = st.columns([3, 1])
    with col_translate:
        translate_button = st.button("🌐 Tłumacz", use_container_width=True, type="primary")
    with col_clear:
        if st.button("🗑️ Wyczyść ", use_container_width=True):
            st.session_state.translation_text = ""
            st.rerun()

    if text and translate_button:
        try:
            with st.spinner('🔄 Tłumaczę tekst...'):
                # Używamy modelu Helsinki-NLP dla tłumaczenia EN→DE
                from transformers import MarianMTModel, MarianTokenizer

                model_name = "Helsinki-NLP/opus-mt-en-de"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                # Tłumaczenie
                translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

            st.success('✅ Tłumaczenie zakończone!')

            # Wyświetlenie tłumaczenia w ładnym boxie
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                        padding: 30px; border-radius: 15px; margin: 20px 0;">
                <h3 style="color: white; margin: 0 0 15px 0;">📝 Przetłumaczony tekst:</h3>
                <p style="font-size: 1.3em; color: white; margin: 0; line-height: 1.6;">{translated_text}</p>
            </div>
            """, unsafe_allow_html=True)

            # Informacja dodatkowa
            st.info('💡 Tłumaczenie wykonane przy użyciu modelu Helsinki-NLP/opus-mt-en-de')

            # Dodaj do historii
            st.session_state.history.append({
                'type': 'Translation EN→DE',
                'input': text[:50] + '...' if len(text) > 50 else text,
                'output': translated_text[:50] + '...' if len(translated_text) > 50 else translated_text,
                'time': time.strftime("%H:%M:%S")
            })

        except Exception as e:
            st.error(f'❌ Wystąpił błąd podczas tłumaczenia: {str(e)}')
            st.warning('⚠️ Upewnij się, że masz zainstalowane wszystkie wymagane biblioteki.')

st.markdown('---')

# Historia
if len(st.session_state.history) > 0:
    with st.expander("📜 Historia operacji", expanded=False):
        st.write(f"**Wykonano operacji: {len(st.session_state.history)}**")

        # Wyświetl ostatnie 5 operacji
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"""
            **{len(st.session_state.history) - i}.** `{item['time']}` - **{item['type']}**
            - Input: _{item['input']}_
            - Output: _{item['output']}_
            """)

        if st.button("🗑️ Wyczyść historię"):
            st.session_state.history = []
            st.rerun()

st.markdown('---')

# Sekcja z danymi (oryginalny kod)
st.header('📊 Przykładowe dane')
st.write('Poniżej prezentujemy przykładowy zbiór danych:')

try:
    df = pd.read_csv("DSP_4.csv", sep=';')
    st.dataframe(df, use_container_width=True)
except FileNotFoundError:
    st.warning('⚠️ Plik DSP_4.csv nie został znaleziony.')
except Exception as e:
    st.error(f'❌ Błąd podczas wczytywania danych: {str(e)}')

st.markdown('---')

# Numer indeksu i informacje
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 👨‍🎓 Informacje o autorze
    - **Numer indeksu:** s27600
    - **Uczelnia:** PJWSTK
    - **Projekt:** Lab05 - Streamlit + Hugging Face
    """)

with col2:
    st.markdown("""
    ### 🔧 Wykorzystane technologie
    - **Streamlit** - framework do tworzenia aplikacji
    - **Hugging Face Transformers** - modele NLP
    - **DistilBERT** - analiza sentymentu
    - **MarianMT** - tłumaczenie maszynowe
    """)

st.markdown('---')
st.caption('Aplikacja NLP - Projekt Lab05 | 2026-05-15')
