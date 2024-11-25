# Chicago Crimes Situation EDA and Prediction

## Overview
This project involves Exploratory Data Analysis (EDA) and prediction modeling on Chicago crime data from 2012 to 2017. The analysis aims to uncover patterns and trends in crime data and predict future occurrences.

## Project Structure

```
└── Chicago Crimes Situation EDA and Prediction
    ├── Chicago_Crimes_DataSet
    │   ├── chicago_community_areas.geojson
    │   └── Chicago_Crimes_2012_to_2017.csv
    ├── Chicago_Economy_DataSet
    │   ├── black.xlsx
    │   ├── citizenship.xlsx    
    │   ├── education.xlsx   
    │   ├── health_insurance.xlsx    
    │   ├── hispanic_latino.xlsx    
    │   ├── median_house_income.xlsx    
    │   ├── noncitizenship.xlsx   
    │   ├── owner_occupied.xlsx
    │   └── poverty.xlsx
    ├── Chicago Crimes Situation EDA and Prediction.html
    ├── Chicago Crimes Situation EDA and Prediction.ipynb
    └── Readme.md
```

## Data
- **Chicago_Crimes_2012_to_2017.csv**: This dataset is too large to be uploaded to GitHub. Please contact [yuxichen749@gmail.com](mailto:yuxichen749@gmail.com) to obtain it.
- **chicago_community_areas.geojson**: GeoJSON file containing community area boundaries in Chicago.
- **Economic Data**: Various Excel files containing economic indicators related to the Chicago area.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd Chicago-Crimes-Situation-EDA-and-Prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Open the Jupyter Notebook `Chicago Crimes Situation EDA and Prediction.ipynb` to explore the data analysis and prediction models.
- The HTML file `Chicago Crimes Situation EDA and Prediction.html` provides a static view of the analysis.

## Results
- The project includes a detailed analysis of crime trends and predictions using machine learning models.
- The results are documented in the Jupyter Notebook and the HTML file.

## Contact
For any questions or issues, please contact meme.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### Key Improvements:
- **Overview**: Provides a brief description of the project.
- **Project Structure**: Clearly outlines the directory structure.
- **Data**: Describes the datasets used and how to obtain them.
- **Installation**: Step-by-step guide to set up the project.
- **Usage**: Instructions on how to use the project files.
- **Results**: Summarizes the outcomes of the analysis.
- **Contact**: Provides contact information for further inquiries.
- **License**: Mentions the licensing information.

This structure ensures that users can easily navigate and understand the project, its purpose, and how to use it.
- jupyter nbconvert --to script "Chicago Crimes Situation EDA and Prediction.ipynb"; pipreqs . --force
