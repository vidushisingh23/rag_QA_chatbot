import pandas as pd

def row_to_summary(row):
    def safe(val, is_money=False):
        if val == "Unknown" or pd.isna(val):
            return "an unknown amount" if is_money else "unknown"
        return f"{int(val)}" if isinstance(val, float) and val.is_integer() else str(val)

    gender = safe(row.get("Gender"))
    married = safe(row.get("Married"))
    marital_status = "married" if married.lower() == "yes" else "unmarried"

    self_employed = row.get("Self_Employed")
    employment_status = "self-employed" if self_employed == "Yes" else "not self-employed"

    return (
        f"Applicant {safe(row.get('Loan_ID'))} is a {gender} and {marital_status} individual "
        f"with {safe(row.get('Dependents'))} dependents, who is {safe(row.get('Education'))} and is "
        f"{employment_status}. They have an income of {safe(row.get('ApplicantIncome'), is_money=True)} "
        f"and a coapplicant income of {safe(row.get('CoapplicantIncome'), is_money=True)}. The loan amount requested "
        f"is {safe(row.get('LoanAmount'), is_money=True)} over a term of {safe(row.get('Loan_Amount_Term'))} months. "
        f"The applicant has a credit history of {safe(row.get('Credit_History'))} and lives in an area "
        f"classified as {safe(row.get('Property_Area'))}. The loan was {'approved' if row['Loan_Status'] == 'Y' else 'rejected'}."
    )

def generate_knowledge_base(csv_path, output_path):
    df = pd.read_csv(csv_path)
    df.fillna("Unknown", inplace=True)
    summaries = df.apply(row_to_summary, axis=1).tolist()
    with open(output_path, "w") as f:
        for line in summaries:
            f.write(line + "\n")
    print(f"Knowledge base saved to {output_path}")

if __name__ == "__main__":
    generate_knowledge_base("train_dataset.csv", "loan_knowledge_base.txt")
