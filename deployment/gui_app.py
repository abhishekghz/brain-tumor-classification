import os
import sys
import shutil
import uuid

import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from deployment.inference import predict_image_bytes
from src.config import CLASS_NAMES, UNLABELED_DIR, LABELED_DIR
from src.continual_train import fine_tune_on_new_data


def _get_admin_password():
    return st.secrets.get("admin_password") or os.getenv("ADMIN_PASSWORD")


def _require_admin(key_suffix):
    admin_password = _get_admin_password()
    if not admin_password:
        st.warning("Set admin password via Streamlit secrets or ADMIN_PASSWORD env var.")
        return False

    entered = st.text_input("Admin password", type="password", key=f"admin_password_{key_suffix}")
    if entered != admin_password:
        st.info("Enter admin password to manage uploads and retrain.")
        return False

    return True


def _save_upload_to_unlabeled(uploaded):
    os.makedirs(UNLABELED_DIR, exist_ok=True)
    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = os.path.join(UNLABELED_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    return path


def _list_unlabeled_files():
    if not os.path.isdir(UNLABELED_DIR):
        return []
    files = [f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort()
    return files


def _count_labeled():
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(LABELED_DIR, cls)
        if os.path.isdir(cls_dir):
            total += len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return total


def main():
    st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
    st.title("Brain Tumor MRI Classifier")
    st.write("Upload an MRI image to get the predicted tumor class and confidence.")

    tab_predict, tab_admin, tab_review = st.tabs(["Predict", "Admin", "Admin Review"])

    with tab_predict:
        uploaded = st.file_uploader(
            "Upload MRI image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded is None:
            st.info("Upload an image to run prediction.")
        else:
            st.image(uploaded, caption="Uploaded image", use_container_width=True)

            if st.button("Predict", type="primary"):
                saved_path = _save_upload_to_unlabeled(uploaded)
                label, conf = predict_image_bytes(uploaded.getvalue())
                if label is None:
                    st.error("Unable to run prediction on this image.")
                    return

                st.success("Prediction complete")
                st.write(f"**Prediction:** {label}")
                st.write(f"**Confidence:** {conf:.4f}")
                st.caption(f"Saved for review: {os.path.basename(saved_path)}")

                if label not in CLASS_NAMES:
                    st.warning("Prediction label is not in configured class list.")

    with tab_admin:
        if not _require_admin("admin"):
            return

        os.makedirs(LABELED_DIR, exist_ok=True)
        st.subheader("Dataset status")
        st.write(f"Unlabeled uploads: {len(_list_unlabeled_files())}")
        st.write(f"Labeled uploads: {_count_labeled()}")

        st.subheader("Label new uploads")
        files = _list_unlabeled_files()
        if not files:
            st.info("No unlabeled uploads available.")
        else:
            selected = st.selectbox("Select an image", files)
            img_path = os.path.join(UNLABELED_DIR, selected)
            st.image(img_path, caption=selected, use_container_width=True)

            label_choice = st.selectbox("Assign label", CLASS_NAMES)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Move to labeled"):
                    target_dir = os.path.join(LABELED_DIR, label_choice)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(img_path, os.path.join(target_dir, selected))
                    st.success(f"Moved to labeled/{label_choice}")
            with col2:
                if st.button("Delete upload"):
                    os.remove(img_path)
                    st.warning("Deleted upload")

        st.subheader("Retrain model")
        if st.button("Fine-tune on new data"):
            with st.spinner("Fine-tuning model... this may take a while"):
                fine_tune_on_new_data()
            st.success("Model updated and saved.")

    with tab_review:
        if not _require_admin("review"):
            return

        st.subheader("Bulk review queue")
        files = _list_unlabeled_files()
        if not files:
            st.info("No unlabeled uploads available.")
            return

        st.caption(f"Total in queue: {len(files)}")
        page_size = st.slider("Items per page", min_value=6, max_value=60, value=12, step=6)
        total_pages = max(1, (len(files) - 1) // page_size + 1)
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

        start = (page - 1) * page_size
        page_files = files[start : start + page_size]

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Select all on page"):
                for name in page_files:
                    st.session_state[f"select_{name}"] = True
        with col_b:
            if st.button("Clear selection on page"):
                for name in page_files:
                    st.session_state[f"select_{name}"] = False

        cols = st.columns(3)
        for idx, name in enumerate(page_files):
            col = cols[idx % 3]
            with col:
                img_path = os.path.join(UNLABELED_DIR, name)
                st.image(img_path, caption=name, use_container_width=True)
                st.checkbox("Select", key=f"select_{name}")

        selected = [name for name in files if st.session_state.get(f"select_{name}")]
        st.caption(f"Selected: {len(selected)}")

        label_choice = st.selectbox("Assign label", CLASS_NAMES, key="bulk_label")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Move selected to labeled", type="primary"):
                if not selected:
                    st.warning("Select at least one image.")
                else:
                    target_dir = os.path.join(LABELED_DIR, label_choice)
                    os.makedirs(target_dir, exist_ok=True)
                    for name in selected:
                        src = os.path.join(UNLABELED_DIR, name)
                        if os.path.exists(src):
                            shutil.move(src, os.path.join(target_dir, name))
                    st.success(f"Moved {len(selected)} images to labeled/{label_choice}")
        with col2:
            if st.button("Delete selected"):
                if not selected:
                    st.warning("Select at least one image.")
                else:
                    for name in selected:
                        src = os.path.join(UNLABELED_DIR, name)
                        if os.path.exists(src):
                            os.remove(src)
                    st.warning(f"Deleted {len(selected)} images")


if __name__ == "__main__":
    main()
