import pandas as pd

# Clean data
# Files
tableA = "adresses-40.csv"
tableB = "XLH_PAT_20240926.csv"

dfA = pd.read_csv(tableA, sep=";", encoding="utf-8")
dfB = pd.read_csv(tableB, sep=",", encoding="utf-8")


previous_addresses = set()

for row in dfA.itertuples(index=False):
    address = (
        str(getattr(row, "numero", "")).strip(),
        str(getattr(row, "rep", "")).strip(),
        str(getattr(row, "nom_voie", "")).strip(),
        str(getattr(row, "code_postal", "")).strip(),
        str(getattr(row, "nom_commune", "")).strip(),
    )

    if address in previous_addresses:
        print(f"Duplicate address found: {address}")
        dfA.drop(dfA[dfA["numero"] == address[0]].index, inplace=True)
    else:
        previous_addresses.add(address)

# Save the cleaned DataFrame to a new CSV file
cleaned_tableA = "tableA.csv"
dfA.to_csv(cleaned_tableA, sep=",", index=False, encoding="utf-8")
print(f"Cleaned data saved to {cleaned_tableA}.")

previous_addresses = set()

for row in dfB.itertuples(index=False):
    address = (
        str(getattr(row, "ADRESSE_3", "")).strip(),
        str(getattr(row, "ADRESSE_4", "")).strip()
    )

    if address in previous_addresses:
        print(f"Duplicate address found: {address}")
        dfB.drop(dfB[dfB["ADRESSE_3"] == address[0]].index, inplace=True)
    else:
        previous_addresses.add(address)

# Save the cleaned DataFrame to a new CSV file
cleaned_tableB = "tableB.csv"
dfB.to_csv(cleaned_tableB, sep=",", index=False, encoding="utf-8")
print(f"Cleaned data saved to {cleaned_tableB}.")