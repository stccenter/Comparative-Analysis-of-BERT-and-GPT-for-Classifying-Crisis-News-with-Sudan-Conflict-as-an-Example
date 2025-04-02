import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv_into_3(
    input_csv,
    output_train_csv="train.csv",
    output_val_csv="val.csv",
    output_test_csv="test.csv",
    test_size=0.1,
    val_ratio=0.15,
    random_state=25
):


    # 1) Read the CSV file using the input_csv parameter
    data = pd.read_csv(input_csv)
    
    # 2) Split off test_size (10% or whatever you set) for test
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # 3) From the remaining train_val_data, split off val_ratio (e.g. 0.15) for validation
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=val_ratio, 
        random_state=random_state
    )
    
    # 4) Save each split to its own CSV
    train_data.to_csv(output_train_csv, index=False)
    val_data.to_csv(output_val_csv, index=False)
    test_data.to_csv(output_test_csv, index=False)
    
    print(f"Train CSV saved to: {output_train_csv} ({len(train_data)} rows)")
    print(f"Validation CSV saved to: {output_val_csv} ({len(val_data)} rows)")
    print(f"Test CSV saved to: {output_test_csv} ({len(test_data)} rows)")

split_csv_into_3("bert_data.csv")