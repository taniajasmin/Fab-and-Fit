# Fab & Fit: AI Food Analyzer & Personalized Meal Planner

A command-line application that leverages Google Gemini AI to analyze meals, extract nutritional information, and generate personalized meal plans. Whether you upload a photo of your meal or enter the details manually, Fab & Fit provides accurate nutrition data and tailored dietary recommendations.

## Features

* **Meal Analysis**: Upload a photo of your meal or enter the ingredients manually to analyze the nutritional content.
* **Accurate Nutrition**: Get detailed breakdowns of calories, macronutrients, and fiber content.
* **Personalized Meal Plans**: Receive a customized meal plan based on your dietary preferences, activity level, and fitness goals.
* **Vegan Support**: Fab & Fit supports vegan dietary preferences and provides relevant meal plan recommendations.

## Required Libraries

The following libraries are required to run Fab & Fit:

* `google-generativeai`: For interacting with the Google Gemini AI model.
* `pillow`: For processing and analyzing images.
* `python-dotenv`: For loading environment variables from a `.env` file.
* `tenacity`: For implementing retry logic for API calls.

## Setup

To get started with Fab & Fit, follow these steps:

1. **Install Required Libraries**: Run `pip install google-generativeai pillow python-dotenv tenacity` to install the required libraries.
2. **Get a Gemini API Key**: Visit https://makersuite.google.com/app/apikey to obtain a Gemini API key.
3. **Create a `.env` File**: Create a `.env` file in the root directory of the project and add your Gemini API key: `GEMINI_API_KEY=your-gemini-api-key-here`.
4. **Run the Script**: Run `python fab_and_fit.py` to start the application.

## Usage

To use Fab & Fit, simply run the script and follow the prompts:

1. Choose to upload a photo of your meal or enter the ingredients manually.
2. Follow the instructions to analyze your meal or generate a personalized meal plan.

## Disclaimer

Fab & Fit is not a substitute for medical advice. The information provided is for informational purposes only, and you should consult a healthcare professional or registered dietitian for personalized dietary recommendations.

## Issues

If you encounter any issues or bugs while using Fab & Fit, please submit an issue on the project's issue tracker.
