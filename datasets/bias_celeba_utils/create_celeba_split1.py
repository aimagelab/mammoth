import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image


def load_dataframes(root_path):
    attributes_file = os.path.join(root_path, 'list_attr_celeba.txt')
    partition_file = os.path.join(root_path, 'list_eval_partition.txt')
    return pd.read_csv(attributes_file, delim_whitespace=True, skiprows=1), pd.read_csv(partition_file, delim_whitespace=True, header=None, names=['image_id', 'partition'])


def assign_labels(attributes_df, partition_df, chunk_attributes, max_per_task, bias_attribute='Male', max_elements_per_group=500):
    final_df = pd.DataFrame()
    assigned_images = set()  # Set to keep track of images that have been assigned

    for i, chunk_attr in enumerate(chunk_attributes):
        target_attr = chunk_attr['attribute']

        for partition in [0, 1, 2]:
            corr_factor = chunk_attr['correlation_factor'] if partition == 0 else 0.5

            for label in [-1, 1]:
                condition = (partition_df['partition'] == partition) & \
                            (attributes_df[target_attr] == label) & \
                            (~attributes_df.index.isin(assigned_images))

                indices_to_assign = attributes_df[condition].index.tolist()
                random.shuffle(indices_to_assign)

                if partition == 0:  # Training partition
                    num_with_bias = int(corr_factor * (max_per_task // 2))
                    num_without_bias = max_per_task // 2 - num_with_bias
                else:  # Validation and Test partitions
                    num_with_bias = min(len([idx for idx in indices_to_assign if attributes_df.loc[idx, bias_attribute] == -label]), max_elements_per_group)
                    num_without_bias = min(len([idx for idx in indices_to_assign if attributes_df.loc[idx, bias_attribute] != -label]), max_elements_per_group)

                    # Ensure gender balance
                    min_count = min(num_with_bias, num_without_bias)
                    num_with_bias = num_without_bias = min_count

                # Check if the image has already been assigned before adding
                indices_with_bias = [idx for idx in indices_to_assign if attributes_df.loc[idx, bias_attribute] == -label and idx not in assigned_images][:num_with_bias]
                indices_without_bias = [idx for idx in indices_to_assign if attributes_df.loc[idx, bias_attribute] != -label and idx not in assigned_images][:num_without_bias]

                all_indices = indices_with_bias + indices_without_bias
                assigned_images.update(all_indices)  # Update the set with newly assigned images

                new_rows = attributes_df.loc[all_indices, ['image_id'] + [attr['attribute'] for attr in chunk_attributes] + [bias_attribute]].copy()
                new_rows['partition'] = partition
                new_rows['Task_Number'] = i
                new_rows['Aligned_With_Bias'] = new_rows.apply(
                    lambda row: 1 if (
                        row[target_attr] == 1 and row[bias_attribute] == -
                        1) or (
                        row[target_attr] == -
                        1 and row[bias_attribute] == 1) else 0,
                    axis=1)
                final_df = pd.concat([final_df, new_rows])

    return final_df


def calculate_statistics(final_df, chunk_attributes, bias_attribute='Male'):
    partition_names = ['Train', 'Validation', 'Test']
    gender_names = ['Female', 'Male']
    gender_colors = {'Female': 'red', 'Male': 'blue'}
    all_attributes = [attr['attribute'] for attr in chunk_attributes]

    for i, chunk_attr in enumerate(chunk_attributes):
        target_attr = chunk_attr['attribute']

        # Filter rows for the current task
        task_df = final_df[final_df['Task_Number'] == i]

        for j, partition in enumerate([0, 1, 2]):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

            # First graph: Bias Ratio
            ax1.set_title(f'Bias Ratio for Task {i} - {partition_names[j]} Split', fontsize=16)
            ax1.set_ylabel('Bias Ratio', fontsize=14)
            ax1.set_xlabel('Group', fontsize=14)
            ax1.set_ylim(0.5, 1)

            labels = []
            ratios = []
            colors = []
            for attr_idx, attr in enumerate(all_attributes):
                for label in [-1, 1]:
                    count_female = task_df[(task_df[attr] == label) & (task_df[bias_attribute] == -1) & (task_df['partition'] == partition)].shape[0]
                    count_male = task_df[(task_df[attr] == label) & (task_df[bias_attribute] == 1) & (task_df['partition'] == partition)].shape[0]
                    total = count_female + count_male
                    max_gender_count = max(count_female, count_male)
                    ratio = max_gender_count / total if total > 0 else 0
                    label_name = attr if label == 1 else f"No {attr}"
                    labels.append(label_name)
                    ratios.append(ratio)

                    dominant_color = gender_colors['Male'] if count_male > count_female else gender_colors['Female']
                    colors.append(dominant_color)

            ax1.bar(labels, ratios, color=colors, alpha=0.7)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.set_axisbelow(True)

            # Second graph: Gender Counts
            ax2.set_title(f'Gender Counts for Task {i} - {partition_names[j]} Split', fontsize=16)
            ax2.set_ylabel('Count', fontsize=14)
            ax2.set_xlabel('Group', fontsize=14)

            counts_female = []
            counts_male = []
            for attr_idx, attr in enumerate(all_attributes):
                for label in [-1, 1]:
                    count_female = task_df[(task_df[attr] == label) & (task_df[bias_attribute] == -1) & (task_df['partition'] == partition)].shape[0]
                    count_male = task_df[(task_df[attr] == label) & (task_df[bias_attribute] == 1) & (task_df['partition'] == partition)].shape[0]
                    counts_female.append(count_female)
                    counts_male.append(count_male)

            bar_width = 0.35
            r1 = range(len(labels))
            r2 = [x + bar_width for x in r1]

            ax2.bar(r1, counts_female, color=gender_colors['Female'], width=bar_width, label='Female', alpha=0.7)
            ax2.bar(r2, counts_male, color=gender_colors['Male'], width=bar_width, label='Male', alpha=0.7)
            ax2.set_xticks([r + bar_width for r in range(len(labels))])
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2.set_axisbelow(True)

            plt.tight_layout()
            os.makedirs('statistics_celeba1', exist_ok=True)
            plt.savefig(f'statistics_celeba1/statistics_task_{i}_{partition_names[j]}.png')
            plt.close(fig)


def save_sample_images(final_df, root_path):
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    attributes = ['Heavy_Makeup', 'Blond_Hair', 'Receding_Hairline', 'Young', 'Wearing_Necklace', 'Bags_Under_Eyes', 'Smiling', 'Eyeglasses']
    bias_values = [-1, 1]  # -1 for Female, 1 for Male
    target_values = [0, 1]
    gender_names = ['Female', 'Male']

    for i, attr in enumerate(attributes):
        for j, bias in enumerate(bias_values):
            for k, target in enumerate(target_values):
                col = j * 2 + k
                sample = final_df[(final_df[attr] == (target * 2 - 1)) & (final_df['Male'] == bias)].sample(1)
                image_path = os.path.join(root_path, 'img_align_celeba', sample['image_id'].values[0])
                img = Image.open(image_path)

                task_number = sample['Task_Number'].values[0]
                gender = gender_names[0] if bias == -1 else gender_names[1]

                axes[i, col].imshow(img)
                axes[i, col].axis('off')
                axes[i, col].set_title(f"Attribute: {attr}\nGender: {gender}\nTarget: {target}")

    plt.tight_layout()
    plt.savefig("griglia.png")


def process_split(root_path):
    len_c_train = 4480
    max_elements_per_group = 100  # Maximum number of elements for each group in validation and test
    epsilon = 0.95

    chunk_attributes = [
        {'attribute': 'Heavy_Makeup', 'correlation_factor': epsilon},
        {'attribute': 'Blond_Hair', 'correlation_factor': epsilon},
        {'attribute': 'Receding_Hairline', 'correlation_factor': epsilon},
        {'attribute': 'Young', 'correlation_factor': epsilon},
        {'attribute': 'Wearing_Necklace', 'correlation_factor': epsilon},
        {'attribute': 'Bags_Under_Eyes', 'correlation_factor': epsilon},
        {'attribute': 'Smiling', 'correlation_factor': epsilon},
        {'attribute': 'Eyeglasses', 'correlation_factor': epsilon},
    ]

    attributes_df, partition_df = load_dataframes(root_path)

    # Reset dell'indice
    attributes_df.reset_index(inplace=True)

    # Rinominare la nuova colonna con il nome 'image_id'
    attributes_df.rename(columns={'index': 'image_id'}, inplace=True)

    final_df = assign_labels(attributes_df, partition_df, chunk_attributes, len_c_train, max_elements_per_group=max_elements_per_group)
    # calculate_statistics(final_df, chunk_attributes)

    # Replace -1 with 0 for specific columns
    attribute_columns = [attr['attribute'] for attr in chunk_attributes]  # Assuming chunk_attributes is a list of dictionaries with 'attribute' keys
    final_df[attribute_columns] = final_df[attribute_columns].replace(-1, 0)
    final_df['Male'] = final_df['Male'].replace(-1, 0)

    # Save final_df to CSV
    # final_df.to_csv(os.path.join(f'biased_celeba1.csv'), index=False)  # _{epsilon}
    final_df.to_csv(os.path.join(root_path, f'biased_celeba1.csv'), index=False)  # _{epsilon}
