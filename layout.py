from dataclasses import dataclass
import streamlit as st


class Dashboard():
    latent_space_selection = ""
    
    def button(self):
        self.first_check = st.button("prova")

