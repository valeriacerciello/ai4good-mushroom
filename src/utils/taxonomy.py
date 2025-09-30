import taxopy
import pandas as pd
import plotly.express as px


def taxid_from_name(name, taxdb, fuzzy=True, score_cutoff=0.7, verbose=False):
    """Wrapper around taxopy.taxid_from_name to prioritize fast,
    non-fuzzy search and fallback to fuzzy search if the exact
    search fails.
    """
    tid = taxopy.taxid_from_name(name, taxdb)
    if tid:
        return tid
    if verbose:
        if fuzzy:
            print(f"❔ Could not find: {name}. Trying fuzzy search with {score_cutoff=}")
        else:
            print(f"❌ Could not find: {name}. Consider using fuzzy=True")
            return tid
    tid = taxopy.taxid_from_name(name, taxdb, fuzzy=fuzzy, score_cutoff=score_cutoff)
    if not tid and verbose:
        print(f"❌ Could not find: {name}. Consider using a larger score_cutoff")
    return tid


def find_highest_parent(edges):
    """Find the highest ancestor of the first encountered species.
    Assumes all species share the same ancestor.
    """
    name = list(edges.keys())[0]
    while name in edges.keys():
        name = edges[name]
    return name


def taxonomic_graph(species_list, taxdb, score_cutoff=0.7, verbose=False):
    # Search for the lineage of each species and convert it to edges
    # in the graph
    # NB: this can take a little while...
    edges = {}
    ranks = {}
    for i_species, name in enumerate(species_list):
        if verbose:
            print(f"[{i_species + 1:>3}/{len(species_list)}] '{name}'")
        try:
            tid = taxid_from_name(
                name,
                taxdb,
                fuzzy=True,
                score_cutoff=score_cutoff,
                verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"Failed to match '{name}'")
            continue
        if not tid:
            if verbose:
                print(f"Failed to match '{name}'")
            continue
        tax = taxopy.Taxon(tid[0], taxdb)
        lineage_dict = tax.rank_name_dictionary
        lineage = list(lineage_dict.values())
        lineage_ranks = list(lineage_dict.keys())

        for i in range(len(lineage) - 1):
            edges[lineage[i]] = lineage[i + 1]
            ranks[lineage[i]] = lineage_ranks[i]

    # Find the highest node in the graph
    root_name = find_highest_parent(edges)
    edges[root_name] = ""
    ranks[root_name] = "root"

    return edges, ranks


def plot_taxonomic_graph(
        edges,
        ranks,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        **kwargs):
    df = pd.DataFrame(list(edges.items()), columns=["name", "parent"])
    df["rank"] = df["name"].map(ranks)
    df["values"] = 1
    df["node_id_str"] = pd.factorize(df["name"])[0].astype(str)

    # Plot sunburst
    fig = px.sunburst(
        df,
        names="name",
        parents="parent",
        values="values",
        color="node_id_str",
        color_discrete_sequence=color_discrete_sequence,
        **kwargs)
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    return fig


if __name__ == "__main__":
    from src.data.dataset import MUSHROOM_CLASS_NAMES, MUSHROOM_TO_TAXOPY

    # Import the taxonomic database, this takes a bit of time
    taxdb = taxopy.TaxDb()

    # Search the species in the database, find their taxonomic lineage,
    # and convert these into graph edges. This takes a bit of time, set
    # verbose=True to track the progress
    taxopy_names = [MUSHROOM_TO_TAXOPY[k] for k in MUSHROOM_CLASS_NAMES]
    edges, ranks = taxonomic_graph(taxopy_names, taxdb, verbose=True)

    # Plot the taxonomic tree
    fig = plot_taxonomic_graph(
        edges,
        ranks,
        width=1000,
        height=1000)

    # Either visualize the plot right here or save it to an HTML file,
    # to be opened with a browser
    # fig.show()
    fig.write_html("media/mushroom_taxonomy.html", include_plotlyjs="cdn")
