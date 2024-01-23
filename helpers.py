import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.utils import resample 

### EDA
def resampling(data):
    majority_class = data[data['Class'] == 0]
    minority_class = data[data['Class'] == 1]
    minority_upsampled = resample(minority_class,
								replace=True,  # Sample with replacement
								n_samples=len(majority_class),  # Match the number of majority class samples
								random_state=42)
    return pd.concat([majority_class, minority_upsampled])

def density_plots_back_to_back(df_a, df_b, col):

	# Create density plots for both columns on the same graph
	sns.kdeplot(df_a[col], label='df_a', shade=True)
	sns.kdeplot(df_b[col], label='df_b', shade=True)

	# Set plot labels and title
	plt.xlabel(f'{col} Values')
	plt.ylabel('Density')
	plt.title('Density Plot of Test Values for df_a and df_b')

	# Show the legend
	plt.legend()

	# Display the plot
	plt.show()