
import streamlit as st
from investtrack_app import InvestTrackPro

def main():
    """Main application entry point"""
    app = InvestTrackPro()
    app.run()

if __name__ == "__main__":
    main()
