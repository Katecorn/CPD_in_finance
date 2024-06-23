# Модуль по построению графа и вычислению его основных статистик
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle
from PyIF import te_compute as te
from tqdm import tqdm
import ruptures as rpt
from community import community_louvain

class GraphBuilder:
    def calculate_diameter(self, G):
        if nx.is_connected(G):
            return nx.diameter(G)
        else:
            diameter = 0
            for subgraph in nx.connected_components(G):
                diameter = max(diameter, nx.diameter(G.subgraph(subgraph)))
            return diameter

    def calculate_graph_statistics(self, G):
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        average_degree = np.mean([d for n, d in G.degree()])
        density = nx.density(G)
        diameter = self.calculate_diameter(G)
        clustering_coefficient = nx.average_clustering(G)
        return {
            "Number of nodes": num_nodes,
            "Number of edges": num_edges,
            "Average degree": average_degree,
            "Density": density,
            "Diameter": diameter,
            "Clustering coefficient": clustering_coefficient
        }

    def build_and_save_te_graph(self, te_data, date, threshold=0.01, path_save='graphs_entropy'):
        os.makedirs(path_save, exist_ok=True)
        G = nx.Graph()

        for i, row in te_data.iterrows():
            if row['TransferEntropy'] > threshold:
                G.add_edge(row['Source'], row['Target'], weight=row['TransferEntropy'])

        if G.number_of_edges() == 0:
            print(f"No connections for threshold {threshold} on {date}")
            return

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', edge_color='grey', font_weight='bold')
        edges = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges)
        plt.title(f"Transfer Entropy Graph on {date}")
        plt.savefig(f'{path_save}/entropy_graph_{date}.png')
        plt.close()

        graph_stats = self.calculate_graph_statistics(G)
        print(graph_stats)

        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        with open(f'{path_save}/entropy_graph_stats_{date_str}.pkl', 'wb') as f:
            pickle.dump({"statistics": graph_stats, "graph": G}, f)

    def build_and_save_corr_graph(self, corr_matrix, date, threshold=0.5, path_save='graphs'):
        os.makedirs(path_save, exist_ok=True)
        G = nx.Graph()

        corr_matrix.set_index(corr_matrix.columns[0], inplace=True)
        
        for crypto1 in corr_matrix.index:
            for crypto2 in corr_matrix.columns:
                if crypto1 != crypto2 and abs(corr_matrix.loc[crypto1, crypto2]) > threshold:
                    G.add_edge(crypto1, crypto2, weight=corr_matrix.loc[crypto1, crypto2])

        if G.number_of_edges() == 0:
            print(f"No connections for threshold {threshold} on {date}")
            return

        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
        edges = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edges)
        plt.title(f"Correlation Graph for {date}")
        plt.savefig(f'{path_save}/graph_{date}.png')
        plt.close()

        graph_stats = self.calculate_graph_statistics(G)
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        with open(f'{path_save}/graph_stats_{date_str}.pkl', 'wb') as f:
            pickle.dump({"statistics": graph_stats, "graph": G}, f)

    def create_daily_te_graphs(self, transfer_entropies, path_save='graphs_entropy'):
        unique_dates = transfer_entropies['Date'].unique()
        for date in tqdm(unique_dates, desc="Processing dates"):
            daily_data = transfer_entropies[transfer_entropies['Date'] == date]
            if daily_data.empty:
                continue
            self.build_and_save_te_graph(daily_data, date, path_save=path_save)

    def create_daily_corr_graphs(self, rolling_correlations, path_save='graphs'):
        unique_dates = rolling_correlations['Date'].unique()
        for date in tqdm(unique_dates, desc="Processing dates"):
            daily_data = rolling_correlations[rolling_correlations['Date'] == date]
            if daily_data.empty:
                continue
            daily_corr_matrix = daily_data.drop(columns=['Date'])
            if daily_corr_matrix.empty or daily_corr_matrix.isnull().values.all():
                print(f"Correlation matrix for {date} is empty or contains only NaN")
                continue
            self.build_and_save_corr_graph(daily_corr_matrix, date, path_save=path_save)

    def plot_graph_statistics(self, stats_dir='graphs'):
        all_stats = {
            "Date": [],
            "Number of nodes": [],
            "Number of edges": [],
            "Average degree": [],
            "Density": [],
            "Diameter": [],
            "Clustering coefficient": []
        }
        
        pkl_files = [file for file in os.listdir(stats_dir) if file.endswith('.pkl')]
        sorted_pkl_files = sorted(pkl_files)
        
        for file in sorted_pkl_files:
            date_str = file.split('_')[-1].replace('.pkl', '')
            date = pd.to_datetime(date_str).date()
            
            with open(os.path.join(stats_dir, file), 'rb') as f:
                data = pickle.load(f)
            
            stats = data["statistics"]
            all_stats["Date"].append(date)
            for key, value in stats.items():
                all_stats[key].append(value)
        
        stats_df = pd.DataFrame(all_stats)
        
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(15, 20))
        axes = axes.flatten()
        
        for i, (key, values) in enumerate(stats_df.drop(columns='Date').items()):
            axes[i].plot(stats_df['Date'], values, marker='o')
            axes[i].set_title(key)
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel(key)
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_graphs(self, graphs, cols=4):
        dates = sorted(graphs.keys())
        num_graphs = len(dates)
        rows = (num_graphs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten()
        
        for i, date in enumerate(dates):
            G = graphs[date]
            pos = nx.spring_layout(G)
            ax = axes[i]
            nx.draw(G, pos, with_labels=True, ax=ax, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
            ax.set_title(f"Graph for {date}")

        for i in range(num_graphs, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()

    def load_graphs_from_files(self, pkl_files, path = 'graphs'):
        graphs = {}
        for file in pkl_files:
            date_str = file.split('_')[-1].replace('.pkl', '')
            date = pd.to_datetime(date_str).date()
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
                graphs[date] = data["graph"]
        return graphs

    def normalize_edge_weights(self, G):
        for u, v, d in G.edges(data=True):
            if 'weight' not in d:
                d['weight'] = 1.0
            if d['weight'] < 0:
                d['weight'] = 0

    def apply_louvain_clustering(self, G):
        self.normalize_edge_weights(G)
        partition = community_louvain.best_partition(G, weight='weight')
        return partition

    def plot_graphs_with_clusters(self, graphs, partitions, cols=4):
        dates = sorted(graphs.keys())
        num_graphs = len(dates)
        rows = (num_graphs + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten()

        for i, date in enumerate(dates):
            G = graphs[date]
            pos = nx.spring_layout(G)
            partition = partitions[date]
            ax = axes[i]

            cmap = plt.get_cmap('viridis')
            unique_clusters = set(partition.values())
            cluster_colors = {cluster: cmap(i / len(unique_clusters)) for i, cluster in enumerate(unique_clusters)}
            colors = [cluster_colors[partition[node]] for node in G.nodes()]

            nx.draw(G, pos, with_labels=True, alpha=0.5, ax=ax, node_size=700, node_color=colors, font_size=10, font_weight='bold')
            ax.set_title(f"Graph for {date}")

        for i in range(num_graphs, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def plot_cluster_changes(self, cluster_df):
        nodes = cluster_df.columns[1:]
        num_nodes = len(nodes)
        cols = 4
        rows = (num_nodes + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
        axes = axes.flatten()

        for i, node in enumerate(nodes):
            ax = axes[i]
            ax.plot(cluster_df['Date'], cluster_df[node], marker='o')
            ax.set_title(f'Cluster Changes for {node}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Cluster')
            ax.grid(True)

        for i in range(num_nodes, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()


# Класс по построению Лаплассиана и реализации расчёта основныз статистик + базовая детекция CPD
class GraphAnalysis:
    def calculate_laplacian_statistics(self, G):
        L = nx.laplacian_matrix(G).toarray()
        eigenvalues = np.linalg.eigvals(L)
        return {
            "Laplacian": L,
            "Eigenvalues": eigenvalues,
            "Second smallest eigenvalue": np.partition(eigenvalues, 1)[1],  
            "Largest eigenvalue": np.max(eigenvalues),
            "Spectral gap": np.partition(eigenvalues, 1)[1] - eigenvalues[0]
        }


# визуализация рассчитанных статистик графа
    def plot_laplacian_statistics(self,laplacian_stats):
        dates = sorted(laplacian_stats.keys())
        second_smallest_eigenvalues = [laplacian_stats[date]["Second smallest eigenvalue"] for date in dates]
        largest_eigenvalues = [laplacian_stats[date]["Largest eigenvalue"] for date in dates]
        
        plt.figure(figsize=(6, 3))
        plt.plot(dates, second_smallest_eigenvalues, marker='o', label='Second smallest eigenvalue')
        plt.plot(dates, largest_eigenvalues, marker='o', label='Largest eigenvalue')
        plt.xlabel('Date')
        plt.ylabel('Eigenvalue')
        plt.title('Laplacian Eigenvalues Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def load_graphs_and_laplacian_stats(self, pkl_files, path):
        graphs = {}
        laplacian_stats = {}
        for file in pkl_files:
            date_str = file.split('_')[-1].replace('.pkl', '')
            date = pd.to_datetime(date_str).date()
            with open(os.path.join(path, file), 'rb') as f:
                data = pickle.load(f)
                G = data["graph"]
                graphs[date] = G
                laplacian_stats[date] = self.calculate_laplacian_statistics(G)
        return graphs, laplacian_stats

    def plot_change_points(self, signal, change_points, title):
        plt.figure(figsize=(6, 3))
        plt.plot(signal, label="Eigenvalue")
        for cp in change_points:
            plt.axvline(cp, color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Eigenvalue')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def apply_cpd(self, laplacian_stats):
        dates = sorted(laplacian_stats.keys())
        second_smallest_eigenvalues = [laplacian_stats[date]["Second smallest eigenvalue"] for date in dates]
        largest_eigenvalues = [laplacian_stats[date]["Largest eigenvalue"] for date in dates]
        spectral_gaps = [laplacian_stats[date]["Spectral gap"] for date in dates]

        second_smallest_eigenvalues = np.array(second_smallest_eigenvalues).astype(float)
        largest_eigenvalues = np.array(largest_eigenvalues).astype(float)
        spectral_gaps = np.array(spectral_gaps).astype(float)

        algo = rpt.Pelt(model="l1").fit(second_smallest_eigenvalues)
        penalty = np.var(second_smallest_eigenvalues)
        change_points_fiedler = algo.predict(pen=penalty)

        algo = rpt.Pelt(model="l1").fit(largest_eigenvalues)
        penalty = np.var(largest_eigenvalues)
        change_points_largest = algo.predict(pen=penalty)

        algo = rpt.Pelt(model="l1").fit(spectral_gaps)
        penalty = np.var(spectral_gaps)
        change_points_spectral_gap = algo.predict(pen=penalty)

        self.plot_change_points(second_smallest_eigenvalues, change_points_fiedler, "Change Points - Second Smallest Eigenvalue")
        self.plot_change_points(largest_eigenvalues, change_points_largest, "Change Points - Largest Eigenvalue")
        self.plot_change_points(spectral_gaps, change_points_spectral_gap, "Change Points - Spectral Gap")

