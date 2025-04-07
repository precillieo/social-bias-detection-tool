import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def create_neural_networks(train_x, test_x, train_y, test_y):
    """
    Create and train different neural network architectures.
    """
    input_dim = train_x.shape[1]
    nn_models = {}
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # 1. Simple Single-Layer Network
    with st.spinner("Training Simple Neural Network..."):
        model1 = Sequential([
            Dense(16, activation='relu', input_dim=input_dim),
            Dense(1, activation='sigmoid')
        ])
        
        model1.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        model1.fit(train_x, train_y, 
                  epochs=10, 
                  batch_size=32, 
                  validation_split=0.2,
                  callbacks=[early_stopping],
                  verbose=0)
        
        nn_models['SimpleNN'] = model1
    
    # 2. Deep Neural Network
    with st.spinner("Training Deep Neural Network..."):
        model2 = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model2.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        model2.fit(train_x, train_y, 
                  epochs=10, 
                  batch_size=32, 
                  validation_split=0.2,
                  callbacks=[early_stopping],
                  verbose=0)
        
        nn_models['DeepNN'] = model2
    
    # 3. Residual-like Network (with skip connections)
    with st.spinner("Training Residual Neural Network..."):
        inputs = Input(shape=(input_dim,))
        x = Dense(32, activation='relu')(inputs)
        
        # First residual block
        block1 = Dense(32, activation='relu')(x)
        block1 = Dropout(0.2)(block1)
        block1 = Dense(32, activation='relu')(block1)
        x = concatenate([x, block1])  # Skip connection
        
        # Second residual block
        block2 = Dense(32, activation='relu')(x)
        block2 = Dropout(0.2)(block2)
        block2 = Dense(32, activation='relu')(block2)
        x = concatenate([x, block2])  # Skip connection
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        residual_model = Model(inputs=inputs, outputs=outputs)
        residual_model.compile(optimizer=Adam(learning_rate=0.001), 
                              loss='binary_crossentropy', 
                              metrics=['accuracy'])
        
        residual_model.fit(train_x, train_y, 
                          epochs=10, 
                          batch_size=32, 
                          validation_split=0.2,
                          callbacks=[early_stopping],
                          verbose=0)
        
        nn_models['ResidualNN'] = residual_model
    
    return nn_models



def calculate_fairness_metrics_manually(y_true, y_pred, protected_attribute, privileged_value):
    """
    Calculate fairness metrics manually.
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected_attribute = np.array(protected_attribute)
    
    # Create masks for privileged and unprivileged groups
    privileged_mask = (protected_attribute == privileged_value)
    unprivileged_mask = ~privileged_mask
    
    # Calculate favorable outcome rates (favorable outcome is 0 - lower expenses)
    priv_selection_rate = np.mean(y_pred[privileged_mask] == 0) if np.sum(privileged_mask) > 0 else 0
    unpriv_selection_rate = np.mean(y_pred[unprivileged_mask] == 0) if np.sum(unprivileged_mask) > 0 else 0
    
    # Calculate Statistical Parity Difference
    spd = unpriv_selection_rate - priv_selection_rate
    
    # Calculate Disparate Impact
    if priv_selection_rate > 0:
        di = unpriv_selection_rate / priv_selection_rate
    else:
        di = float('inf') if unpriv_selection_rate > 0 else 1.0
    
    # Calculate Equal Opportunity Difference
    priv_positives = np.sum(y_true[privileged_mask] == 0)
    unpriv_positives = np.sum(y_true[unprivileged_mask] == 0)
    
    if priv_positives > 0:
        priv_tpr = np.sum((y_pred[privileged_mask] == 0) & (y_true[privileged_mask] == 0)) / priv_positives
    else:
        priv_tpr = 0
        
    if unpriv_positives > 0:
        unpriv_tpr = np.sum((y_pred[unprivileged_mask] == 0) & (y_true[unprivileged_mask] == 0)) / unpriv_positives
    else:
        unpriv_tpr = 0
    
    eod = unpriv_tpr - priv_tpr
    
    # Calculate Average Odds Difference
    priv_negatives = np.sum(y_true[privileged_mask] == 1)
    unpriv_negatives = np.sum(y_true[unprivileged_mask] == 1)
    
    if priv_negatives > 0:
        priv_fpr = np.sum((y_pred[privileged_mask] == 0) & (y_true[privileged_mask] == 1)) / priv_negatives
    else:
        priv_fpr = 0
        
    if unpriv_negatives > 0:
        unpriv_fpr = np.sum((y_pred[unprivileged_mask] == 0) & (y_true[unprivileged_mask] == 1)) / unpriv_negatives
    else:
        unpriv_fpr = 0
    
    tpr_diff = unpriv_tpr - priv_tpr
    fpr_diff = unpriv_fpr - priv_fpr
    aod = (tpr_diff + fpr_diff) / 2
    
    return {
        'Statistical Parity Difference': spd,
        'Disparate Impact': di,
        'Equal Opportunity Difference': eod,
        'Average Odds Difference': aod
    }


def main():
    st.title("Social Bias Detection in Medical Research Data")

    st.write("""
             ## Upload your dataset for processing.
             This tool helps researchers identify both data and algorithm level biases in their datasets.
             """)
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Data loading
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.write(df.head())
        st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Data preprocessing
        st.write("### Data Preprocessing")

        missing_columns = df.columns[df.isnull().any()].tolist()
        if missing_columns:
            st.write(f"Found {len(missing_columns)} columns with missing values")
            
            if st.button("Clean Missing Values"):
                original_shape = df.shape
                df = df.dropna(axis=1, thresh=0.5*len(df))
                remaining_missing = df.columns[df.isnull().any()].tolist()
                if remaining_missing:
                    df = df.dropna(axis=1)

                st.write(f"Dropped {original_shape[1] - df.shape[1]} columns with missing values")
                st.success("Missing values cleaned!")
                
                # Store in session state
                st.session_state['cleaned_df'] = df
        else:
            st.write("No missing values found in the dataset.")
            # Store in session state
            st.session_state['cleaned_df'] = df
            
        # Only proceed if data is cleaned
        if 'cleaned_df' in st.session_state:
            df = st.session_state['cleaned_df']
            
            # Create age features if birthdate exists
            if 'BIRTHDATE' in df.columns:
                current_year = datetime.now().year
                df['AGE'] = df['BIRTHDATE'].apply(
                    lambda x: current_year - int(x.split('-')[0]) if pd.notna(x) else np.nan
                )
                    
                # Age groups
                bins = [0, 18, 35, 50, 65, 120]
                labels = ['0-18', '19-35', '36-50', '51-65', '65+']
                df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels)
                
                st.write("Created AGE and AGE_GROUP features from BIRTHDATE")
            
            # Select columns for analysis
            st.write("### Select Analysis Parameters")
            
            # Suggest protected attributes
            all_columns = df.columns.tolist()
            suggested_protected = [col for col in all_columns if col.upper() in 
                                ['RACE', 'ETHNICITY', 'GENDER', 'AGE_GROUP']]
            
            protected_attrs = st.multiselect(
                "Select protected attributes (e.g., race, gender):",
                options=all_columns,
                default=suggested_protected if suggested_protected else []
            )

            # Suggest target variables
            suggested_targets = [col for col in all_columns if 'EXPENSE' in col.upper() 
                               or 'COST' in col.upper() or 'COVERAGE' in col.upper()]
            
            target_var = st.selectbox(
                "Select target variable:",
                options=all_columns,
                index=all_columns.index(suggested_targets[0]) if suggested_targets else 0
            )

            if protected_attrs and target_var:
                # Create binary target if numeric
                if pd.api.types.is_numeric_dtype(df[target_var]):
                    median_value = df[target_var].median()
                    binary_target = f"HIGH_{target_var}"
                    df[binary_target] = (df[target_var] > median_value).astype(int)
                    st.write(f"Created binary target: {binary_target} (1 if > {median_value:.2f}, 0 otherwise)")
                    target_var = binary_target
                
                # Store processed dataframe
                st.session_state['processed_df'] = df
                
                # Data-level bias detection
                st.write("### Data-Level Bias Detection")
                
                tabs = st.tabs(["Class Imbalance", "Statistical Parity", "Demographic Disparity"])
                
                # Tab 1: Class Imbalance
                with tabs[0]:
                    st.write("#### Class Imbalance Analysis")
                    
                    for attr in protected_attrs:
                        st.write(f"**Distribution of {attr}:**")
                        
                        # Calculate value counts and percentages
                        value_counts = df[attr].value_counts()
                        percentages = df[attr].value_counts(normalize=True) * 100
                        
                        # Create a DataFrame for display
                        imbalance_df = pd.DataFrame({
                            'Count': value_counts,
                            'Percentage': percentages
                        })
                        imbalance_df['Percentage'] = imbalance_df['Percentage'].map('{:.1f}%'.format)
                        
                        # Display the table
                        st.write(imbalance_df)
                        
                        # Flag underrepresented groups
                        underrepresented = percentages[percentages < 10].index.tolist()
                        if underrepresented:
                            st.warning(f"⚠️ **Underrepresented groups** (<10%): {', '.join(map(str, underrepresented))}")
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 5))
                        bars = sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                        ax.set_title(f'Distribution of {attr}')
                        ax.set_ylabel('Count')
                        
                        # Add percentage labels
                        for i, p in enumerate(bars.patches):
                            percentage = percentages.iloc[i]
                            bars.annotate(f'{percentage:.1f}%', 
                                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                                         ha = 'center', va = 'bottom', 
                                         xytext = (0, 5), textcoords = 'offset points')
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Tab 2: Statistical Parity Difference
                with tabs[1]:
                    st.write("#### Statistical Parity Difference (SPD)")
                    st.write("Measures whether privileged groups receive favorable outcomes more often than unprivileged groups.")
                    
                    for attr in protected_attrs:
                        st.write(f"**SPD Analysis for {attr}:**")
                        
                        # Calculate SPD
                        groups = df[attr].unique()
                        reference_group = df[attr].value_counts().idxmax()
                        reference_rate = df[df[attr] == reference_group][target_var].mean()
                        
                        # Create results table
                        results_data = []
                        
                        for group in groups:
                            if group != reference_group:
                                group_count = df[df[attr] == group].shape[0]
                                group_percent = (group_count / len(df)) * 100
                                group_rate = df[df[attr] == group][target_var].mean()
                                spd = group_rate - reference_rate
                                
                                results_data.append({
                                    'Group': group,
                                    'Count': group_count,
                                    'Percentage': f"{group_percent:.1f}%",
                                    'Favorable Rate': f"{group_rate:.4f}",
                                    'SPD': f"{spd:.4f}",
                                    'Assessment': "✓ Minimal" if abs(spd) < 0.05 else 
                                                 "⚠️ Small" if abs(spd) < 0.10 else 
                                                 "⚠️ Substantial"
                                })
                        
                        # Display reference group info
                        st.write(f"Reference group: **{reference_group}** (favorable outcome rate: {reference_rate:.4f})")
                        
                        # Display results table
                        if results_data:
                            results_df = pd.DataFrame(results_data)
                            st.write(results_df)
                        else:
                            st.write("No comparison groups available.")
                
                # Tab 3: Demographic Disparity
                with tabs[2]:
                    st.write("#### Demographic Disparity (DD)")
                    st.write("Identifies groups disproportionately facing negative outcomes.")
                    
                    for attr in protected_attrs:
                        st.write(f"**DD Analysis for {attr}:**")
                        
                        # Calculate demographic disparity
                        total_count = len(df)
                        group_counts = df[attr].value_counts()
                        group_proportions = group_counts / total_count
                        
                        unfavorable_df = df[df[target_var] == 1]  # Unfavorable outcome is 1
                        unfavorable_count = len(unfavorable_df)
                        unfavorable_group_counts = unfavorable_df[attr].value_counts()
                        unfavorable_group_proportions = unfavorable_group_counts / unfavorable_count
                        
                        # Create results table
                        results_data = []
                        
                        for group in group_proportions.index:
                            overall_prop = group_proportions[group]
                            unfavorable_prop = unfavorable_group_proportions.get(group, 0)
                            disparity = unfavorable_prop - overall_prop
                            disparity_ratio = unfavorable_prop / overall_prop if overall_prop > 0 else float('inf')
                            
                            results_data.append({
                                'Group': group,
                                'Population %': f"{overall_prop:.4f}",
                                'Unfavorable Outcome %': f"{unfavorable_prop:.4f}",
                                'Disparity': f"{disparity:.4f}",
                                'Assessment': "✓ Minimal" if abs(disparity) < 0.05 else 
                                             "⚠️ Moderate" if abs(disparity) < 0.10 else 
                                             "⚠️ Substantial"
                            })
                        
                        # Display results table
                        results_df = pd.DataFrame(results_data)
                        st.write(results_df)
                        
                        # Visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        x = np.arange(len(group_proportions.index))
                        width = 0.35
                        
                        ax.bar(x - width/2, group_proportions, width, label='Population Representation')
                        
                        # Handle missing groups in unfavorable outcomes
                        unfav_props = [unfavorable_group_proportions.get(group, 0) for group in group_proportions.index]
                        ax.bar(x + width/2, unfav_props, width, label='Unfavorable Outcome Representation')
                        
                        ax.set_ylabel('Proportion')
                        ax.set_title(f'Demographic Disparity - {attr}')
                        ax.set_xticks(x)
                        ax.set_xticklabels(group_proportions.index)
                        ax.legend()
                        
                        # Add disparity values as text
                        for i, group in enumerate(group_proportions.index):
                            disparity = unfavorable_group_proportions.get(group, 0) - group_proportions[group]
                            color = 'green' if abs(disparity) < 0.05 else 'orange' if abs(disparity) < 0.10 else 'red'
                            ax.annotate(f"Disparity: {disparity:.3f}", 
                                       xy=(i, max(group_proportions[group], unfavorable_group_proportions.get(group, 0)) + 0.02),
                                       ha='center', va='bottom', color=color)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                # Algorithmic bias detection placeholder
                st.write("### Algorithmic Bias Detection")
                if st.button("Run Algorithmic Bias Analysis"):
                    with st.spinner("Preparing data for modeling..."):
                        # Create a copy for preprocessing
                        model_df = df.copy()
                        
                        # Encode categorical variables
                        categorical_columns = model_df.select_dtypes(include=['object', 'category']).columns.tolist()
                        encoders = {}
                        
                        for column in categorical_columns:
                            le = LabelEncoder()
                            model_df[column] = le.fit_transform(model_df[column])
                            encoders[column] = le
                        
                        # Store race encoding for fairness evaluation
                        race_mapping = dict(zip(encoders['RACE'].classes_, range(len(encoders['RACE'].classes_))))
                        st.write(f"Race encoding: {race_mapping}")
                        
                        # Prepare features and target
                        cols_to_drop = ['Id'] if 'Id' in model_df.columns else []
                        if 'HEALTHCARE_EXPENSES' in model_df.columns:
                            cols_to_drop.append('HEALTHCARE_EXPENSES')
                        if 'LOW_COVERAGE' in model_df.columns:
                            cols_to_drop.append('LOW_COVERAGE')
                        
                        X = model_df.drop(cols_to_drop + [target_var], axis=1)
                        y = model_df[target_var]
                        
                        # Scale numerical features
                        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                        scaler = StandardScaler()
                        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        st.write(f"Training data shape: {X_train.shape}")
                        st.write(f"Testing data shape: {X_test.shape}")
                    
                    # Train traditional models with LazyPredict
                    with st.spinner("Training multiple models with LazyPredict..."):
                        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
                        models_df, _ = clf.fit(X_train, X_test, y_train, y_test)
                        
                        st.write("#### Model Performance")
                        st.write(models_df)
                        
                        # Get top 5 models
                        num_top_models = min(5, len(models_df))
                        top_model_names = models_df.index[:num_top_models].tolist()
                        st.write(f"Top {num_top_models} models: {top_model_names}")
                        
                        # Extract models
                        top_models = {}
                        for model_name in top_model_names:
                            if model_name in clf.models:
                                pipeline = clf.models[model_name]
                                classifier = pipeline.named_steps.get('classifier', None)
                                
                                if classifier:
                                    model_class = classifier.__class__
                                    top_models[model_name] = model_class()
                    
                    # Train neural network models
                    with st.spinner("Training neural network models..."):
                        nn_models = create_neural_networks(X_train, X_test, y_train, y_test)
                        st.write("Neural network models trained successfully")
                    
                    with st.spinner("Evaluating fairness metrics for all models..."):
                        # Combine traditional and neural network models
                        all_models = {**top_models, **nn_models}
                        
                        # Store results
                        fairness_results = {}
                        model_accuracies = {}
                        
                        # Get the white encoded value for fairness calculations
                        white_encoded = race_mapping.get('white', 0)
                        
                        # Create tabs for results
                        model_tabs = st.tabs(["Model Performance", "Fairness Metrics", "Accuracy vs. Fairness"])
                        
                        with model_tabs[0]:
                            st.write("#### Model Accuracy Comparison")
                            
                            # Train and evaluate each model
                            results_data = []
                            
                            for model_name, model in all_models.items():
                                try:
                                    # Train traditional models (neural networks are already trained)
                                    if model_name not in nn_models:
                                        model.fit(X_train, y_train)
                                    
                                    # Get predictions
                                    if model_name in nn_models:
                                        y_pred_prob = model.predict(X_test)
                                        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                                    else:
                                        y_pred = model.predict(X_test)
                                    
                                    # Calculate accuracy
                                    accuracy = (y_pred == y_test).mean()
                                    model_accuracies[model_name] = accuracy
                                    
                                    # Evaluate fairness metrics
                                    fairness_metrics = calculate_fairness_metrics_manually(
                                        y_test, 
                                        y_pred, 
                                        X_test['RACE'],
                                        white_encoded
                                    )
                                    
                                    fairness_results[model_name] = fairness_metrics
                                    
                                    # Add to results
                                    results_data.append({
                                        'Model': model_name,
                                        'Accuracy': f"{accuracy:.4f}",
                                        'Type': 'Neural Network' if model_name in nn_models else 'Traditional'
                                    })
                                    
                                except Exception as e:
                                    st.error(f"Error evaluating {model_name}: {str(e)}")
                            
                            # Display results table
                            results_df = pd.DataFrame(results_data)
                            st.write(results_df)
                            
                            # Visualization
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Group by model type
                            nn_models_acc = [float(row['Accuracy']) for _, row in results_df[results_df['Type'] == 'Neural Network'].iterrows()]
                            trad_models_acc = [float(row['Accuracy']) for _, row in results_df[results_df['Type'] == 'Traditional'].iterrows()]
                            nn_names = results_df[results_df['Type'] == 'Neural Network']['Model'].tolist()
                            trad_names = results_df[results_df['Type'] == 'Traditional']['Model'].tolist()
                            
                            x = np.arange(max(len(nn_models_acc), len(trad_models_acc)))
                            width = 0.35
                            
                            # Plot bars
                            if nn_models_acc:
                                nn_bars = ax.bar(x[:len(nn_models_acc)] - width/2, nn_models_acc, width, label='Neural Networks')
                            if trad_models_acc:
                                trad_bars = ax.bar(x[:len(trad_models_acc)] + width/2, trad_models_acc, width, label='Traditional Models')
                            
                            ax.set_ylabel('Accuracy')
                            ax.set_title('Model Accuracy Comparison')
                            ax.set_xticks(x)
                            ax.set_ylabel('Accuracy')
                            ax.set_title('Model Accuracy Comparison')
                            ax.set_xticks(x[:max(len(nn_names), len(trad_names))])
                            
                            # Set x-tick labels
                            if len(nn_names) > len(trad_names):
                                ax.set_xticklabels(nn_names)
                            else:
                                ax.set_xticklabels(trad_names)
                            
                            ax.legend()
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        with model_tabs[1]:
                            st.write("#### Fairness Metrics Comparison")
                            
                            # Create fairness metrics table
                            fairness_data = []
                            
                            for model_name, metrics in fairness_results.items():
                                fairness_data.append({
                                    'Model': model_name,
                                    'Statistical Parity Difference': f"{metrics['Statistical Parity Difference']:.4f}",
                                    'Equal Opportunity Difference': f"{metrics['Equal Opportunity Difference']:.4f}",
                                    'Average Odds Difference': f"{metrics['Average Odds Difference']:.4f}",
                                    'Type': 'Neural Network' if model_name in nn_models else 'Traditional'
                                })
                            
                            # Display fairness table
                            fairness_df = pd.DataFrame(fairness_data)
                            st.write(fairness_df)
                            
                            # Visualization for each fairness metric
                            metrics_to_plot = ['Statistical Parity Difference', 'Equal Opportunity Difference', 'Average Odds Difference']
                            
                            for metric in metrics_to_plot:
                                st.write(f"**{metric} by Model Type**")
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                # Extract values, convert to float and take absolute value
                                nn_values = [abs(float(row[metric])) for _, row in fairness_df[fairness_df['Type'] == 'Neural Network'].iterrows()]
                                trad_values = [abs(float(row[metric])) for _, row in fairness_df[fairness_df['Type'] == 'Traditional'].iterrows()]
                                
                                x = np.arange(max(len(nn_values), len(trad_values)))
                                width = 0.35
                                
                                # Plot bars
                                if nn_values:
                                    nn_bars = ax.bar(x[:len(nn_values)] - width/2, nn_values, width, label='Neural Networks')
                                if trad_values:
                                    trad_bars = ax.bar(x[:len(trad_values)] + width/2, trad_values, width, label='Traditional Models')
                                
                                # Add a reference line for "fair" threshold
                                ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
                                ax.text(0, 0.1, 'Fairness Threshold (0.1)', va='bottom', ha='left', color='r')
                                
                                ax.set_ylabel(f'|{metric}|')
                                ax.set_title(f'Absolute {metric} by Model Type')
                                ax.set_xticks(x[:max(len(nn_names), len(trad_names))])
                                
                                # Set x-tick labels
                                if len(nn_names) > len(trad_names):
                                    ax.set_xticklabels(nn_names)
                                else:
                                    ax.set_xticklabels(trad_names)
                                
                                ax.legend()
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)
                        
                        with model_tabs[2]:
                            st.write("#### Accuracy vs. Fairness Trade-offs")
                            
                            # Extract metrics for visualization
                            model_names = list(fairness_results.keys())
                            accuracies = [model_accuracies[model] for model in model_names]
                            spd_values = [abs(fairness_results[model]['Statistical Parity Difference']) for model in model_names]
                            eod_values = [abs(fairness_results[model]['Equal Opportunity Difference']) for model in model_names]
                            model_types = ['Neural Network' if model in nn_models else 'Traditional' for model in model_names]
                            
                            # Create scatter plot
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Use different markers for neural networks vs traditional models
                            for i, model_type in enumerate(set(model_types)):
                                indices = [j for j, t in enumerate(model_types) if t == model_type]
                                marker = 'o' if model_type == 'Neural Network' else '^'
                                
                                scatter = ax.scatter(
                                    [accuracies[j] for j in indices],
                                    [spd_values[j] for j in indices],
                                    c=[eod_values[j] for j in indices],
                                    marker=marker,
                                    s=100,
                                    alpha=0.7,
                                    label=model_type
                                )
                            
                            # Add labels for each point
                            for i, model in enumerate(model_names):
                                ax.annotate(
                                    model,
                                    (accuracies[i], spd_values[i]),
                                    fontsize=9,
                                    ha='center',
                                    va='bottom'
                                )
                            
                            plt.colorbar(scatter, label='|Equal Opportunity Difference|')
                            plt.xlabel('Accuracy')
                            plt.ylabel('|Statistical Parity Difference|')
                            plt.title('Accuracy vs. Fairness Trade-offs for Different Models')
                            plt.grid(True, linestyle='--', alpha=0.7)
                            plt.legend()
                            
                            # Add a reference line for "fair" SPD threshold
                            plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
                            plt.text(min(accuracies), 0.1, 'Fairness Threshold (SPD=0.1)', 
                                    va='bottom', ha='left', color='r')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Create combined score and ranking
                            st.write("#### Model Ranking (Balancing Accuracy and Fairness)")
                            
                            # Normalize metrics
                            max_accuracy = max(accuracies)
                            max_spd = max(spd_values) if max(spd_values) > 0 else 1
                            max_eod = max(eod_values) if max(eod_values) > 0 else 1
                            
                            # Calculate combined scores
                            combined_scores = []
                            for i, model in enumerate(model_names):
                                # Weight accuracy and fairness equally (0.5 each)
                                accuracy_score = accuracies[i] / max_accuracy
                                fairness_score = 1 - ((spd_values[i] / max_spd + eod_values[i] / max_eod) / 2)
                                combined_score = 0.5 * accuracy_score + 0.5 * fairness_score
                                combined_scores.append(combined_score)
                            
                            # Create ranking dataframe
                            ranking_data = []
                            for i, model in enumerate(model_names):
                                ranking_data.append({
                                    'Model': model,
                                    'Type': 'Neural Network' if model in nn_models else 'Traditional',
                                    'Accuracy': f"{accuracies[i]:.4f}",
                                    'Statistical Parity Difference': f"{fairness_results[model]['Statistical Parity Difference']:.4f}",
                                    'Equal Opportunity Difference': f"{fairness_results[model]['Equal Opportunity Difference']:.4f}",
                                    'Combined Score': f"{combined_scores[i]:.4f}"
                                })
                            
                            # Sort by combined score
                            ranking_df = pd.DataFrame(ranking_data)
                            ranking_df = ranking_df.sort_values('Combined Score', ascending=False).reset_index(drop=True)
                            
                            # Display ranking
                            st.write(ranking_df)
                            
                            # Highlight best model
                            best_model = ranking_df.iloc[0]['Model']
                            best_score = ranking_df.iloc[0]['Combined Score']
                            st.success(f"**Best model balancing accuracy and fairness:** {best_model} (Combined Score: {best_score})")
                            
                            # Recommendations
                            st.write("#### Recommendations")
                            
                            # Check if any model has high accuracy but poor fairness
                            high_acc_poor_fairness = []
                            for i, model in enumerate(model_names):
                                if accuracies[i] > 0.9 and (spd_values[i] > 0.1 or eod_values[i] > 0.1):
                                    high_acc_poor_fairness.append(model)
                            
                            if high_acc_poor_fairness:
                                st.warning(f"⚠️ The following models have high accuracy but poor fairness: {', '.join(high_acc_poor_fairness)}")
                                st.write("Consider using a more balanced model even if it has slightly lower accuracy.")
                            
                            # # Check if neural networks perform differently than traditional models
                            # nn_avg_fairness = np.mean([spd_values[i] for i, model in enumerate(model_names) if model in nn_models]) if nn_models else 0
                            # trad_avg_fairness = np.mean([spd_values[i] for i, model in enumerate(model_names) if model not in nn_models]) if len(model_names) > len(nn_models) else 0
                            
                            # if abs(nn_avg_fairness - trad_avg_fairness) > 0.05:
                            #     better_type = "Neural Networks" if nn_avg_fairness < trad_avg_fairness else "Traditional Models"
                            #     st.info(f"📊 {better_type} tend to have better fairness metrics on this dataset.")

if __name__ == "__main__":
    main()



