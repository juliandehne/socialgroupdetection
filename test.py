from socialgroupdetection import SGA

sga = SGA(gwdg_server=True)
response = sga.get_social_groups(
    texts_or_text=[
        "The teleworker brings only his or her work tools offices are generally equipped and pays for access to the office?",
        "I stress this because every time we propose a step in this direction the Government and DISY are breaking their clothes that the economy of the country will be destroyed"
    ],
    embedding_based_filtering=True)
print(response)
