import json
import os
order1= [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
19, 65, 23, 94, 89, 78, 76, 45, 69, 12,
73, 75, 53, 61, 14, 48, 56, 93, 8, 13,
66, 59, 95, 96, 28, 92, 46, 98, 22, 54,
85, 20, 34, 27, 38, 86, 40, 4, 29, 26,
71, 49, 72, 33, 88, 31, 36, 17, 99, 41,
80, 67, 64, 5, 35, 90, 21, 70, 9, 2,
62,7, 16, 6, 60, 3, 81, 32, 74, 25,
30, 79, 83, 57, 18, 55, 50, 77, 84, 10,
1, 43, 39, 63, 37, 24, 42, 47, 11, 82]

order2 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
 66, 2, 22, 14, 53, 28, 39, 35, 60, 48,
 95, 9, 88, 82, 71, 65, 24, 67, 32, 84,
74, 1, 77, 59, 19, 31, 75, 6, 94, 37,
 18, 45, 73, 61, 13, 98, 99, 46, 81, 17,
78, 34, 62, 20, 36, 42, 92, 93, 76, 72,
 21, 26, 49, 23, 47, 70, 83, 33, 40, 38,
86, 57, 30, 7, 63, 50, 8, 55, 69, 89,
 64, 10, 11, 12, 96, 79, 29, 3, 41, 85,
   43, 16, 80, 5, 25, 56, 27, 90, 4, 54]

order3 = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
            48, 10, 90, 17, 54, 92, 31, 73, 40, 71,
            29, 76, 60, 37, 14, 11, 77, 1, 53, 81,
            98, 63, 70, 59, 2, 45, 33, 85, 88, 22,
            9, 95, 86, 16, 41, 8, 43, 47, 74, 93,
            64, 50, 75, 62, 3, 89, 56, 34, 39, 84,
            20, 78, 7, 12, 57, 42, 21, 55, 35, 65,
            32, 72, 66, 6, 99, 94, 25, 18, 27, 46,
            61, 36, 23, 79, 69, 49, 96, 28, 83, 19,
            67, 26, 38, 80, 13, 30, 24, 82, 4, 5]
    

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")


def get_values_from_json(json_file, indexes):
    # Open and read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Get the keys of the dictionary
    keys = list(data.keys())

    # Retrieve values based on the list of indexes
    selected_values = [data[keys[i]] for i in indexes if i < len(keys)]

    return selected_values

# Example usage
json_file = '/home/shivank2/gokhale_user/shivanand/icarl-pytorch/data/concept_sets/decider_cifar100_concepts.json'  # Replace with your actual JSON file path
# indexes = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15]  # List of indexes you want to retrieve values from
exp3 = [87,  0, 52, 58, 44, 91, 68, 97, 51, 15,
                        94, 92, 10, 72, 49, 78, 61, 14,  8, 86,
                        84, 96, 18, 24, 32, 45, 88, 11,  4, 67,
                        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
                        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
                        1, 28,  6, 46, 62, 82, 53,  9, 31, 75,
                         38, 63, 33, 74, 27, 22, 36,  3, 16, 21,
                         60, 19, 70, 90, 89, 43,  5, 42, 65, 76,
                         40, 30, 23, 85,  2, 95, 56, 48, 71, 64,
                         98, 13, 99,  7, 34, 55, 54, 26, 35, 39]
n_exp = 10

for i in range(len(order3)//n_exp):
    curr_lst=[]
    curr_lst.append(order3[0:(i+1)*n_exp])
    print("Current list:", curr_lst)
    selected_values = get_values_from_json(json_file, curr_lst[0])
    # print("Selected values:", selected_values)
    file_path = f"/home/shivank2/gokhale_user/shivanand/icarl-pytorch/data/concept_sets/order2_cifar100_concepts/new_exp{i+1}_filtered_new.txt"
    folder_path=file_path.split("/")[0:-1]
    folder_path="/".join(folder_path)
    # print("File path:", "/".join(file_path))
    # ensure_folder_exists("/".join(file_path))   
    ensure_folder_exists(folder_path)
    with open(file_path, "a") as file:
        for sublist in selected_values:
            for element in sublist:
                file.write(f"{element}\n")









# print(f"All elements have been appended to {file_path}.")

# selected_values = get_values_from_json(json_file, exp3)
# print("Selected values:", selected_values)

# # List of lists
# list_of_lists = selected_values

# # Filepath to the .txt file
# file_path = "/home/shivank2/gokhale_user/shivanand/icarl-pytorch/data/concept_sets/order1_cifar100_concepts/new_exp10_filtered_new.txt"

# # Append each element of all lists to the file
# with open(file_path, "a") as file:
#     for sublist in list_of_lists:
#         for element in sublist:
#             file.write(f"{element}\n")

# print(f"All elements have been appended to {file_path}.")
