from cProfile import label
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .attack {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load trained model and preprocessing objects."""
    models_dict = {}
    try:
        # Core model
        models_dict['xgb'] = joblib.load("low_model.pkl")

        # Preprocessing
        models_dict['scaler'] = joblib.load("scaler.pkl")
        models_dict['le'] = joblib.load("label_encoder.pkl")
        models_dict['dummy_columns'] = joblib.load("dummy_columns.pkl")
        models_dict['service_counts'] = joblib.load("service_freq_map.pkl")

        st.success("✅ All models and preprocessing objects loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")

    return models_dict



@st.cache_data
def get_feature_names():
    """Get the feature names used in training"""
    # Based on the notebook analysis, these are the top features
    return [
        'duration', 'src_bytes', 'dst_bytes', 'logged_in', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count'
    ]

@st.cache_data
def get_complete_column_names():
    """Get the complete column names for NSL-KDD dataset"""
    return [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
        "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
        "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "attack", "level"
    ]

@st.cache_data
def get_categorical_options():
    """Get categorical feature options"""
    return {
        'protocol_type': ['tcp', 'udp', 'icmp'],
        'service': ['http', 'smtp', 'finger', 'auth', 'telnet', 'ftp', 'private', 'pop_3', 'ftp_data', 'ntp_u', 'other', 'ecr_i', 'time', 'domain', 'ssh', 'name', 'whois', 'mtp', 'gopher', 'rje', 'vmnet', 'daytime', 'link', 'supdup', 'uucp', 'netstat', 'kshell', 'echo', 'discard', 'systat', 'csnet_ns', 'iso_tsap', 'hostnames', 'exec', 'login', 'shell', 'printer', 'efs', 'courier', 'uucp_path', 'netbios_ns', 'netbios_dgm', 'netbios_ssn', 'sql_net', 'X11', 'urh_i', 'urp_i', 'pm_dump', 'tftp_u', 'red_i', 'harvest'],
        'flag': ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH', 'SH']
    }

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def preprocess_input(df_input, scaler, feature_list, label_encoder=None):
    """
    Preprocess a single row or batch of input.
    Returns:
        X_scaled: np.ndarray
        y_true: int or None
    """
    import pandas as pd

    if isinstance(df_input, dict):
        df_input = pd.DataFrame([df_input])
    
    # Handle categorical columns
    for col in ['protocol_type','service','flag']:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype('category').cat.codes

    y_true = None
    if 'attack' in df_input.columns:
        # Create binary column exactly like in training
        df_input['attack_binary'] = df_input['attack'].apply(lambda x: "normal" if x == "normal" else "attack")

        le = LabelEncoder()
        df_input['attack_binary'] = df_input['attack'].apply(lambda x: 0 if x == "normal" else 1)
        y_true = df_input['attack_binary']
        df_input = df_input.drop(columns=['attack', 'attack_binary'])  # drop before scaling
        
    # Keep only features used in training
    df_input = df_input.reindex(columns=feature_list, fill_value=0)
    
    # Scale
    X_scaled = scaler.transform(df_input)

    return X_scaled, y_true




from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

def predict_attack(models: dict, X_scaled, y_true=None):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values

    xgb_model = models["xgb"]

    # Predict numeric classes
    preds = xgb_model.predict(X_scaled)

    # Flip 0 ↔ 1 if model is inverted
    preds_flipped = 1 - preds.astype(int)  

    probs = xgb_model.predict_proba(X_scaled)

    results_list = []
    for i in range(len(preds_flipped)):
        results_list.append({
            "index": i,
            "prediction": int(preds_flipped[i]),  # 0=normal, 1=attack after flip
            "normal_prob": float(probs[i][1]),    # swap prob columns if needed
            "attack_prob": float(probs[i][0]),
            "status": "Normal" if preds_flipped[i] == 0 else "Attack"
        })

    results_df = pd.DataFrame(results_list)

    metrics = None
    if y_true is not None:
        y_true_arr = np.array(y_true)
        y_pred_labels = preds_flipped
        metrics = {
            "accuracy": accuracy_score(y_true_arr, y_pred_labels),
            "precision": precision_score(y_true_arr, y_pred_labels, zero_division=0),
            "recall": recall_score(y_true_arr, y_pred_labels, zero_division=0),
            "f1": f1_score(y_true_arr, y_pred_labels, zero_division=0)
        }

    return results_df, metrics


def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Intrusion Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading ensemble model..."):
        models = load_model()  # dictionary of models
    
    if models is None or len(models) == 0:
        st.error("Failed to load model. Please check if the model files are in the correct location.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Batch Upload", "Model Information", "About"]
    )
    
    if page == "Batch Upload":
        batch_upload_page(models)  # pass the dictionary
    elif page == "Model Information":
        model_info_page(models)
    elif page == "About":
        about_page()


def prediction_page(models):
    """Main prediction interface"""
    st.header("🔍 Network Traffic Analysis")
    st.markdown("Enter network traffic features to detect potential intrusions.")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Connection Features")
        
        # Basic features
        duration = st.number_input("Duration (seconds)", min_value=0, max_value=100000, value=0)
        protocol_type = st.selectbox("Protocol Type", get_categorical_options()['protocol_type'])
        service = st.selectbox("Service", get_categorical_options()['service'])
        flag = st.selectbox("Flag", get_categorical_options()['flag'])
        
        src_bytes = st.number_input("Source Bytes", min_value=0, max_value=1000000000, value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=1000000000, value=0)
        
        # Connection features
        land = st.selectbox("Land (same host/port)", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0, max_value=100, value=0)
        urgent = st.number_input("Urgent", min_value=0, max_value=100, value=0)
        hot = st.number_input("Hot", min_value=0, max_value=100, value=0)
        
    with col2:
        st.subheader("Advanced Features")
        
        # Authentication features
        num_failed_logins = st.number_input("Number of Failed Logins", min_value=0, max_value=100, value=0)
        logged_in = st.selectbox("Logged In", [0, 1])
        num_compromised = st.number_input("Number Compromised", min_value=0, max_value=100, value=0)
        root_shell = st.selectbox("Root Shell", [0, 1])
        su_attempted = st.selectbox("SU Attempted", [0, 1])
        
        # File and shell features
        num_root = st.number_input("Number of Root Accesses", min_value=0, max_value=100, value=0)
        num_file_creations = st.number_input("Number of File Creations", min_value=0, max_value=100, value=0)
        num_shells = st.number_input("Number of Shells", min_value=0, max_value=100, value=0)
        num_access_files = st.number_input("Number of Access Files", min_value=0, max_value=100, value=0)
        
    # Additional features in a third column
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Connection Statistics")
        
        count = st.number_input("Count", min_value=0, max_value=1000, value=0)
        srv_count = st.number_input("Service Count", min_value=0, max_value=1000, value=0)
        serror_rate = st.slider("Service Error Rate", 0.0, 1.0, 0.0, 0.01)
        srv_serror_rate = st.slider("Service Service Error Rate", 0.0, 1.0, 0.0, 0.01)
        rerror_rate = st.slider("Response Error Rate", 0.0, 1.0, 0.0, 0.01)
        srv_rerror_rate = st.slider("Service Response Error Rate", 0.0, 1.0, 0.0, 0.01)
        
    with col4:
        st.subheader("Host Statistics")
        
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 0.0, 0.01)
        diff_srv_rate = st.slider("Different Service Rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_count = st.number_input("Destination Host Count", min_value=0, max_value=1000, value=0)
        dst_host_srv_count = st.number_input("Destination Host Service Count", min_value=0, max_value=1000, value=0)
        dst_host_same_srv_rate = st.slider("Destination Host Same Service Rate", 0.0, 1.0, 0.0, 0.01)
        dst_host_diff_srv_rate = st.slider("Destination Host Different Service Rate", 0.0, 1.0, 0.0, 0.01)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Analyze Traffic", type="primary", use_container_width=True):
        with st.spinner("Analyzing network traffic..."):
            # Collect all input data
            input_data = {
                'duration': duration,
                'protocol_type': protocol_type,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': land,
                'wrong_fragment': wrong_fragment,
                'urgent': urgent,
                'hot': hot,
                'num_failed_logins': num_failed_logins,
                'logged_in': logged_in,
                'num_compromised': num_compromised,
                'root_shell': root_shell,
                'su_attempted': su_attempted,
                'num_root': num_root,
                'num_file_creations': num_file_creations,
                'num_shells': num_shells,
                'num_access_files': num_access_files,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'srv_serror_rate': srv_serror_rate,
                'rerror_rate': rerror_rate,
                'srv_rerror_rate': srv_rerror_rate,
                'same_srv_rate': same_srv_rate,
                'diff_srv_rate': diff_srv_rate,
                'dst_host_count': dst_host_count,
                'dst_host_srv_count': dst_host_srv_count,
                'dst_host_same_srv_rate': dst_host_same_srv_rate,
                'dst_host_diff_srv_rate': dst_host_diff_srv_rate
            }
            

            # Then call predict_attack correctly
            result = predict_attack(models, input_data)
            
            if result:
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Prediction result
                prediction = result['prediction']
                probability = result['probability']
                
                if prediction == 0:
                    st.markdown(
                        f'<div class="prediction-box normal">'
                        f'<h3>✅ Normal Traffic Detected</h3>'
                        f'<p>Confidence: {probability[0]:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box attack">'
                        f'<h3>🚨 Attack Detected!</h3>'
                        f'<p>Confidence: {probability[1]:.2%}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                # Display probability breakdown
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Probability", f"{probability[0]:.2%}")
                with col2:
                    st.metric("Attack Probability", f"{probability[1]:.2%}")
                
                # Feature importance visualization
                st.subheader("Feature Analysis")
                processed_features = result['processed_features']
                feature_importance = processed_features.iloc[0].abs().sort_values(ascending=False)
                
                st.bar_chart(feature_importance.head(10))

def batch_upload_page(models):
    """Batch upload and prediction interface"""
    import streamlit as st
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    st.header("📁 Batch File Upload")
    st.markdown("Upload a CSV or TXT file with network traffic data for batch analysis.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        help="Upload a CSV or TXT file with network traffic data"
    )

    if uploaded_file is not None:
        expected_cols = get_complete_column_names()

        # Try reading first line to guess if headers exist
        preview_df = pd.read_csv(uploaded_file, nrows=5, header=None)
        has_headers = all(isinstance(x, str) for x in preview_df.iloc[0].values)
        uploaded_file.seek(0)

        if has_headers:
            df = pd.read_csv(uploaded_file)
            if list(df.columns) != expected_cols:
                st.info("ℹ️ Headers detected but incorrect. Assigning default column names...")
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = expected_cols
        else:
            df = pd.read_csv(uploaded_file, header=None)
            df.columns = expected_cols

        df = df[expected_cols]
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Prediction options
        st.subheader("🎯 Prediction Options")
        col1, col2 = st.columns(2)
        with col1:
            prediction_type = st.selectbox(
                "Prediction Type",
                ["All Records", "Sample (First 10)", "Custom Range"]
            )
        with col2:
            start_idx, end_idx = 0, 0
            if prediction_type == "Custom Range":
                start_idx = st.number_input("Start Index", min_value=0, max_value=len(df)-1, value=0)
                end_idx = st.number_input("End Index", min_value=start_idx, max_value=len(df)-1,
                                          value=min(start_idx+9, len(df)-1))

        # Process button
        if st.button("🚀 Process Predictions", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                # Select subset
                if prediction_type == "All Records":
                    data_to_process = df.copy()
                elif prediction_type == "Sample (First 10)":
                    data_to_process = df.head(10).copy()
                else:
                    data_to_process = df.iloc[start_idx:end_idx+1].copy()

                # ----------------------
                # Preprocess features and extract y_true if exists
                feature_list = get_feature_names()  # your 15-feature list
                scaler = models["scaler"]
                label_encoder = models.get("label_encoder", None)

                X_scaled_list = []
                y_true_list = []
                for _, row in data_to_process.iterrows():
                    X_scaled, y_true = preprocess_input(
                        row.to_dict(),
                        scaler=scaler,
                        feature_list=feature_list,
                        label_encoder=label_encoder
                    )
                    X_scaled_list.append(X_scaled[0])  # X_scaled is 2D np.ndarray
                    y_true_list.append(y_true)

                import numpy as np
                
                X_scaled_batch = np.array(X_scaled_list)
                # Only include non-None y_true
                if any(yt is not None for yt in y_true_list):
                    y_true_batch = np.array([yt for yt in y_true_list])
                else:
                    y_true_batch = None

                # Run predictions
                results_df, metrics = predict_attack(models, X_scaled_batch, y_true_batch)

                # Show results
                st.subheader("📈 Batch Prediction Results")
                st.dataframe(results_df, use_container_width=True)

                # Show metrics if available
                if results_df is not None:
                    st.subheader("✅ Cross-Validation Metrics")
                    # Convert attack column to binary 0/1
                    y_true_bin = np.array([0 if x == "normal" else 1 for x in data_to_process["attack"]])
                    y_pred_bin = results_df['status'].apply(lambda x: 0 if x.lower() == "normal" else 1).values

                    acc = accuracy_score(y_true_bin, y_pred_bin)
                    prec = precision_score(y_true_bin, y_pred_bin)
                    rec = recall_score(y_true_bin, y_pred_bin)
                    f1 = f1_score(y_true_bin, y_pred_bin)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1: st.metric("Accuracy", f"{acc:.4f}")
                    with col2: st.metric("Precision", f"{prec:.4f}")
                    with col3: st.metric("Recall", f"{rec:.4f}")
                    with col4: st.metric("F1 Score", f"{f1:.4f}")

                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"prediction_results_{uploaded_file.name}",
                    mime="text/csv"
                )




def model_info_page(model):
    """Display model information"""
    st.header("🤖 Model Information")
    
    st.subheader("Ensemble Model")
    
    # Display model type
    st.write(f"**Model**: {type(model).__name__}")
    st.write("**File**: ong.pkl")
    st.write("**Type**: Ensemble Model (combines multiple algorithms)")
    
    st.subheader("Feature Information")
    st.write("The model uses the following key features:")
    
    features = get_feature_names()
    for i, feature in enumerate(features, 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("Complete Dataset Structure")
    st.write("The NSL-KDD dataset contains the following columns:")
    
    complete_features = get_complete_column_names()
    for i, feature in enumerate(complete_features, 1):
        st.write(f"{i}. {feature}")
    
    st.subheader("Model Performance")
    st.info("""
    The model was trained on the NSL-KDD dataset and achieves high accuracy 
    in detecting various types of network intrusions including:
    - DoS (Denial of Service)
    - Probe attacks
    - R2L (Remote to Local)
    - U2R (User to Root)
    """)

def about_page():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### Intrusion Detection System
    
    This web application provides real-time network traffic analysis to detect potential security intrusions.
    
    #### Features:
    - **Real-time Analysis**: Analyze network traffic patterns instantly
    - **Multiple Models**: Uses ensemble of machine learning models for better accuracy
    - **User-friendly Interface**: Simple form-based input for network features
    - **Detailed Results**: Provides confidence scores and feature analysis
    
    #### How it works:
    1. Enter network traffic features in the form
    2. Click "Analyze Traffic" to process the data
    3. View the prediction results and confidence scores
    4. Examine feature importance for insights
    
    #### Dataset:
    - Trained on NSL-KDD dataset
    - Contains various types of network attacks
    - Includes both normal and malicious traffic patterns
    
    #### Technology Stack:
    - **Backend**: Python, Scikit-learn, TensorFlow
    - **Frontend**: Streamlit
    - **Model**: Ensemble Model (ong.pkl)
    """)

if __name__ == "__main__":
    main()
