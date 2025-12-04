from sklearn.decomposition import PCA
import datamapplot as dmp
import matplotlib as plt

def reduce_embeddings_to_2d(embeddings):
    pca = PCA(n_components=2, random_state=42)
    umap_2d = pca.fit_transform(embeddings)   # your saved 7D embeddings
    return umap_2d

# function to extract the first four words of the "Representation" column
def extract_first_n_words(representations, top_n=4):
    representations = str(representations)
    return ' '.join(representations.strip("[").strip("]").split(",")[:top_n]).replace("'", "").replace('"', '').strip().replace("  ", ", ")

def label_topic(row, selected_topics):
    if row['Topic'] in selected_topics and not pd.isna(row['Top_Words']):
        return f"{row['Rank']}: {row['Top_Words']}"
    else:
        return "Unlabelled"
    
def visualize_docs(model_name, umap_embedding, sorted_topic_info, doc_info, selected_topics=range(0,30), use_english_label=True): # plot the top 30 topics by default
    # first reduce embeddings to 2D
    umap_2d = reduce_embeddings_to_2d(umap_embedding)
    # apply the extract_first_n_words function to the "Representation" column, n = 4
    if use_english_label:
        sorted_topic_info["Top_Words"] = sorted_topic_info["Translation"].apply(lambda x: extract_first_n_words(x, top_n=4))
    else:
        sorted_topic_info["Top_Words"] = sorted_topic_info["Representation"].apply(lambda x: extract_first_n_words(x, top_n=4))
    # merge doc_info with sorted_topic_info to get the "Top_Words" column
    labeled_doc_info = doc_info[["sentence", "Topic"]].merge(sorted_topic_info, left_on='Topic', right_on='Topic', how='left')
    # for selected topics, 
    labeled_doc_info['plot_labels'] = labeled_doc_info.apply(lambda row: label_topic(row, selected_topics), axis=1)
    plot_labels = labeled_doc_info['plot_labels'].tolist()

    fig = dmp.create_plot(
        umap_2d,
        plot_labels,
        marker_type="o",
        title=f"Top 30 Topics Identified by {model_name.upper()} Model",
        sub_title="A 2-dimensional data map of sentences from Dutch IBD patient messages",
        label_over_points=True,
        dynamic_label_size=True,
        max_font_size=50,
        min_font_size=6,
    )

    # save the figure
    if use_english_label:
        plot_path = f"/home/jzhang/mijnidbcoachnlp/results/{model_name}/{model_name}_{len(selected_topics)}_topics_datamap_english.png"
    else:
        plot_path = f"/home/jzhang/mijnidbcoachnlp/results/{model_name}/{model_name}_{len(selected_topics)}_topics_datamap.png"
    plt.pyplot.savefig(plot_path, dpi=300)