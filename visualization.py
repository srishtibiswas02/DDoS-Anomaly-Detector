import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_visualizations(data):
    ############# PIE CHART ##########################
    class_distribution = data['Predictions'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(class_distribution, labels=["Attack","Benign"], autopct='%1.1f%%', startangle=140, colors=['lightcoral', 'turquoise'])
    plt.title('Distribution of Predictions (Attack vs. Normal Traffic)')
    plt.axis('equal') 
    plt.savefig(os.path.join('static', 'visualization1.png'))


    ############# UNSTACKED BAR CHART  ##########################
    attack_traffic = data[data['Predictions'] == 1]
    benign_traffic = data[data['Predictions'] == 0]
    attack_ports = attack_traffic['Destination Port'].value_counts().head(10)
    benign_ports = benign_traffic['Destination Port'].value_counts().head(10)
    port_counts = pd.DataFrame({'Attack': attack_ports, 'Benign': benign_ports}).fillna(0)
    port_counts.plot(kind='bar', stacked=False, color=['#FF6347', '#4CAF50'], figsize=(10, 6))
    plt.title('Top 10 Destination Ports for Attack and Benign Traffic')
    plt.xlabel('Destination Port')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('static', 'visualization2.png'))


    #HISTOGRAM
    plt.figure(figsize=(10, 6))
    plt.hist(data['Flow Duration'], bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Flow Duration")
    plt.xlabel("Flow Duration (milliseconds)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join('static', 'visualization3.png'))

    traffic_packets = data.groupby('Predictions')[['Total Fwd Packets', 'Total Backward Packets']].sum()
    traffic_packets.index = ['Benign', 'Attack']

    # Plot stacked bar chart
    traffic_packets.plot(kind='bar', stacked=True, color=['#4CAF50', '#FF6347'], figsize=(8, 6))
    plt.title('Total Packets (Forward and Backward) by Traffic Type')
    plt.xlabel('Traffic Type')
    plt.ylabel('Total Packets')
    plt.legend(['Total Fwd Packets', 'Total Backward Packets'])
    plt.savefig(os.path.join('static','visualization4.png'))
    plt.close()
