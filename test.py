from socialgroupdetection import SGA

sga = SGA(gwdg_server=True)
response = sga.get_social_groups(
    texts_or_text=[
        "The teleworker brings only his or her work tools offices are generally equipped and pays for access to the office?",
        "I stress this because every time we propose a step in this direction the Government and DISY are breaking their clothes that the economy of the country will be destroyed",
        "it can be a stressful work environment for women and men",
        "Money should follow the child so that competition between schools can raise quality"
    ],
    embedding_based_filtering=False,
    filter_type="svm", as_dataframe=True)
print(response.explicit)
