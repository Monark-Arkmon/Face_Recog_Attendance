import pickle
import numpy as np
import os

def load_face_data():
    """Load the existing face data and names from pickle files."""
    try:
        with open('attendance_system/backend/data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        with open('attendance_system/backend/data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        return names, faces_data
    except FileNotFoundError:
        print("Data files not found!")
        return [], None

def save_face_data(names, faces_data):
    """Save the updated face data and names to pickle files."""
    with open('attendance_system/backend/data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('attendance_system/backend/data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
    print("Data files updated successfully!")

def remove_person(name_to_remove):
    """Remove all instances of a person from the face recognition system."""
    names, faces_data = load_face_data()
    if not names or faces_data is None:
        return False
    
    indices_to_remove = [i for i, name in enumerate(names) if name == name_to_remove]
    
    if not indices_to_remove:
        print(f"No data found for {name_to_remove}")
        return False
    
    for index in sorted(indices_to_remove, reverse=True):
        names.pop(index)
        faces_data = np.delete(faces_data, index, axis=0)
    
    save_face_data(names, faces_data)
    print(f"Removed {len(indices_to_remove)} entries for {name_to_remove}")
    return True

def list_registered_people():
    """List all unique names in the system."""
    names, _ = load_face_data()
    if names:
        unique_names = sorted(set(names))
        print("\nRegistered people:")
        for name in unique_names:
            count = names.count(name)
            print(f"- {name} ({count} face samples)")
    else:
        print("No registered people found")

if __name__ == "__main__":
    while True:
        print("\n=== Face Recognition Data Management ===")
        print("1. List registered people")
        print("2. Remove person")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            list_registered_people()
            
        elif choice == "2":
            list_registered_people()
            name = input("\nEnter the name to remove: ")
            confirm = input(f"Are you sure you want to remove {name}? (yes/no): ")
            if confirm.lower() == 'yes':
                remove_person(name)
            else:
                print("Operation cancelled")
                
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")