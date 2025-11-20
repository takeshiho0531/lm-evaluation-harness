import faiss
import numpy as np
import os
import sys

# 1st index (0-1)
index1_path = "/share/gupta/compactds/index_IVFPQ.100000000.768.65536.256.s0_e479.faiss"
# meta1_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/index_IVFPQ.100000000.768.65536.256.faiss.s0_e1.meta"
# pos_ids1_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/passage_pos_id_array.s0_e1.npy"
# filenames1_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/passage_filenames.s0_e1.npy"

# 2nd index (2-3)
index2_path = "/share/gupta/compactds/index_IVFPQ.100000000.768.65536.256.s480_e959.faiss"
# meta2_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/index_IVFPQ.100000000.768.65536.256.faiss.s2_e3.meta"
# pos_ids2_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/passage_pos_id_array.s2_e3.npy"
# filenames2_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/index_IVFPQ/passage_filenames.s2_e3.npy"

# output path
merged_index_path = "/share/gupta/compactds/merged_index.s0_e959.faiss"
# merged_meta_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/merged_index.s0_e3.faiss.meta"
# merged_pos_ids_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/merged_pos_ids.s0_e3.npy"
# merged_filenames_path = "/share5/akiho.kawada/compactds/datastores/compactds/embeddings/merged_filenames.s0_e3.npy"

# ----------------------------------------------------

print("--- Loading Index 1 files ---")
try:
    index1 = faiss.read_index(index1_path)
    # with open(meta1_path, 'rb') as f:
    #     meta1 = np.load(f)
    # with open(pos_ids1_path, 'rb') as f:
    #     pos_ids1 = np.load(f)
    # with open(filenames1_path, 'rb') as f:
    #     filenames1 = np.load(f, allow_pickle=True)
except FileNotFoundError as e:
    print(f"Error loading Index 1 files: {e}. Please check your paths.")
    sys.exit(1)

print("\n--- Loading Index 2 files ---")
try:
    index2 = faiss.read_index(index2_path)
    # with open(meta2_path, 'rb') as f:
    #     meta2 = np.load(f)
    # with open(pos_ids2_path, 'rb') as f:
    #     pos_ids2 = np.load(f)
    # with open(filenames2_path, 'rb') as f:
    #     filenames2 = np.load(f, allow_pickle=True)
except FileNotFoundError as e:
    print(f"Error loading Index 2 files: {e}. Please check your paths.")
    sys.exit(1)

print("\n--- Index 1 Stats ---")
print(f"Index 1 ntotal: {index1.ntotal}")
# print(f"Meta 1 length: {len(meta1)}")
# print(f"Pos_ids 1 length: {len(pos_ids1)}")
# print(f"Filenames 1 length: {len(filenames1)}")

print("\n--- Index 2 Stats ---")
print(f"Index 2 ntotal: {index2.ntotal}")
# print(f"Meta 2 length: {len(meta2)}")
# print(f"Pos_ids 2 length: {len(pos_ids2)}")
# print(f"Filenames 2 length: {len(filenames2)}")


# --- 1. merging index ---
print("\nMerging index 2 into index 1...")
offset = index1.ntotal
index1.merge_from(index2, offset)
print(f"New merged index ntotal: {index1.ntotal}")

# --- 2. merging passage_filenames ---
# print("Merging passage_filenames files...")
# filenames_merged = np.concatenate((filenames1, filenames2))
# print(f"New merged filenames length: {len(filenames_merged)}")

# --- 3. merging passage_pos_ids ---
# print("Merging passage_pos_ids files...")
# pos_ids_merged = np.concatenate((pos_ids1, pos_ids2))
# print(f"New merged pos_ids length: {len(pos_ids_merged)}")

# --- 4. merging metadata (with offset) ---
# print("Merging metadata files (with offset)...")
# # The IDs in meta2 are increased by an offset equal to the number of files in filenames1
# meta_offset = len(filenames1)
# print(f"Applying offset to meta2: {meta_offset}")
# meta2_adjusted = meta2 + meta_offset
# meta_merged = np.concatenate((meta1, meta2_adjusted))
# print(f"New merged meta length: {len(meta_merged)}")


# --- 5. valification ---
# print("\nVerifying merged files...")
# try:
#     assert index1.ntotal == len(meta_merged)
#     print("Verification 1/2 successful: Index ntotal matches meta length.")
    
#     # Since index_id is also used as the index for pos_ids_merged, the two must have the same length
#     assert index1.ntotal == len(pos_ids_merged)
#     print("Verification 2/2 successful: Index ntotal matches pos_ids length.")

# except AssertionError as e:
#     print(f"!!! CRITICAL WARNING !!!")
#     print(f"Mismatch detected: {e}")
#     if index1.ntotal != len(meta_merged):
#         print(f"  Index ntotal ({index1.ntotal}) != Meta length ({len(meta_merged)})")
#     if index1.ntotal != len(pos_ids_merged):
#         print(f"  Index ntotal ({index1.ntotal}) != Pos_ids length ({len(pos_ids_merged)})")
#     print("Do not proceed if this error occurs.")
#     sys.exit(1)

# --- 6. save ---
print(f"\nSaving merged index to: {merged_index_path}")
faiss.write_index(index1, merged_index_path)

# print(f"Saving merged meta to: {merged_meta_path}")
# with open(merged_meta_path, 'wb') as f:
#     np.save(f, meta_merged)

# print(f"Saving merged pos_ids to: {merged_pos_ids_path}")
# with open(merged_pos_ids_path, 'wb') as f:
#     np.save(f, pos_ids_merged)

# print(f"Saving merged filenames to: {merged_filenames_path}")
# with open(merged_filenames_path, 'wb') as f:
#     np.save(f, filenames_merged)

print("\nDone. All merged files (index, meta, pos_ids, filenames) have been saved.")