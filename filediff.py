import os
import sys

def compare_folders(folder1, folder2):
    # Get the set of all file names in both folders
    files_in_folder1 = set(os.listdir(folder1))
    files_in_folder2 = set(os.listdir(folder2))

    # Find files that are only in folder1
    only_in_folder1 = files_in_folder1 - files_in_folder2
    # Find files that are only in folder2
    only_in_folder2 = files_in_folder2 - files_in_folder1
    # Find files that are in both folders
    in_both_folders = files_in_folder1 & files_in_folder2

    return only_in_folder1, only_in_folder2, in_both_folders, len(files_in_folder1), len(files_in_folder2)

def main(folder1, folder2):
    only_in_folder1, only_in_folder2, in_both_folders, count_folder1, count_folder2 = compare_folders(folder1, folder2)

    print(f"\nTotal files in folder 1: {count_folder1}")
    print(f"Total files in folder 2: {count_folder2}")

    print("\nFiles only in folder 1:")
    for file in only_in_folder1:
        print(file)

    print("\nFiles only in folder 2:")
    for file in only_in_folder2:
        print(file)

    print("\nFiles that are in both folders:")
    for file in in_both_folders:
        print(file)
    print("\nPercent diff: "+ str(count_folder1/count_folder2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_folders.py <folder1> <folder2>")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    main(folder1, folder2)
