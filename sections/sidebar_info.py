import streamlit as st


def sidebar_info():
    
    with st.sidebar:
        
        st.title("text-to-icpc2")

        st.subheader("Descrição do projeto")
        st.write(
            "Este é um demo do modelo text-to-icpc2, onde é possível classificar um diagnóstico num código ICPC-2"
        )
        
        st.write("")
        
        st.write(
            "O modelo foi treinado com dados de diagnósticos de saúde e códigos ICPC-2 e correspondencias com o ICD-10"
        )
        st.write(
            "O modelo foi treinado com a biblioteca Hugging Face *transformers* com base no modelo pré-treinado **bert-base-uncased** e está disponível em [https://huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)"
        )

        st.write("")

        st.write(
            "Este projeto foi desenvolvido por Diogo Carapito com o apoio de bolsa de inviestigação da [AICIB](https://aicib.pt/) e [APMGF](https://apmgf.pt/) no âmbito do internato médico de MGF"
        )
        st.write(
            "Github do projeto: [https://github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)"
        )

        