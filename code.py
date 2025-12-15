#  CARGA DE DATOS

internet_path = "/datasets/final_provider/internet.csv"
personal_path = "/datasets/final_provider/personal.csv"
phone_path = "/datasets/final_provider/phone.csv"
contract_path = "/datasets/final_provider/contract.csv"

internet = pd.read_csv(internet_path)
personal = pd.read_csv(personal_path)
phone = pd.read_csv(phone_path)
contract = pd.read_csv(contract_path)

datasets = {
    "internet": internet,
    "personal": personal,
    "phone": phone,
    "contract": contract
}

#  VISTA GENERAL

print("VISTA GENERAL DE ARCHIVOS")

for name, df in datasets.items():
    print(f"--- {name.upper()} ---")
    print("Shape:", df.shape)
    print("Columnas:", df.columns.tolist())
    print(df.head(), "\n")

#  INFO Y TIPOS DE DATO

print("INFO Y TIPOS DE DATOS")

for name, df in datasets.items():
    print(f"--- {name.upper()} ---")
    print(df.info(), "\n")

#  DESCRIPCIÓN ESTADÍSTICA

print("DESCRIPCIÓN ESTADÍSTICA")

for name, df in datasets.items():
    print(f"--- {name.upper()} ---")
    print(df.describe(include="all"), "\n")

#  NULOS Y DUPLICADOS

print("NULOS Y DUPLICADOS")

for name, df in datasets.items():
    print(f"--- {name.upper()} ---")
    print("Nulos por columna:")
    print(df.isna().sum())
    print("Duplicados totales:", df.duplicated().sum())
    print("\n")

# LIMPIEZA TotalCharges (contract)


print("LIMPIEZA TotalCharges")

print("Tipo original TotalCharges:", contract["TotalCharges"].dtype)

# Convertir a numérico
contract["TotalCharges"] = pd.to_numeric(
    contract["TotalCharges"], errors="coerce")

print("Tipo nuevo TotalCharges:", contract["TotalCharges"].dtype)
print("Nulos en TotalCharges tras conversión:",
      contract["TotalCharges"].isna().sum())

# Para ver filas problemáticas:
print("\nEjemplo filas con TotalCharges nulo:")
print(contract[contract["TotalCharges"].isna()].head())

#  GRÁFICAS

print("GRÁFICAS")


# Distribución de Churn
if "Churn" in contract.columns:
    contract["Churn"].value_counts().plot(
        kind="bar", title="Distribución de Churn")
    plt.show()

# Histograma de MonthlyCharges
if "MonthlyCharges" in contract.columns:
    contract["MonthlyCharges"].plot(
        kind="hist", bins=30, title="MonthlyCharges")
    plt.show()

# Histograma de TotalCharges
if "TotalCharges" in contract.columns:
    contract["TotalCharges"].dropna().plot(
        kind="hist", bins=30, title="TotalCharges")
    plt.show()

# TARGET
# Target = 1 si EndDate == "No" (cliente sigue)
# Target = 0 si EndDate != "No" (cliente canceló)
contract["Target_EndDate_No"] = (contract["EndDate"] == "No").astype(int)

# FECHAS
contract["BeginDate"] = pd.to_datetime(contract["BeginDate"], errors="coerce")
end_dates = pd.to_datetime(
    contract["EndDate"].where(contract["EndDate"] != "No"),
    errors="coerce"
)

snapshot_date = pd.concat([contract["BeginDate"], end_dates]).max()

contract["tenure_days"] = (snapshot_date - contract["BeginDate"]).dt.days
contract["begin_year"] = contract["BeginDate"].dt.year
contract["begin_month"] = contract["BeginDate"].dt.month

# LIMPIEZA NUMÉRICA
contract["TotalCharges"] = pd.to_numeric(
    contract["TotalCharges"], errors="coerce")

# MERGE DE LOS 4 CSV
df = contract.merge(personal, on="customerID", how="left") \
             .merge(phone, on="customerID", how="left") \
             .merge(internet, on="customerID", how="left")

# LIMPIEZA DE NULOS
target = "Target_EndDate_No"
drop_cols = ["customerID", "BeginDate", "EndDate", target]

# Categóricas: "Unknown"
for c in df.select_dtypes(include="object").columns:
    if c not in ["customerID", "BeginDate", "EndDate", target]:
        df[c] = df[c].fillna("Unknown")

# Numéricas: mediana
for c in df.select_dtypes(exclude="object").columns:
    if c != target:
        df[c] = df[c].fillna(df[c].median())

# FEATURES / TARGET
X = df.drop(columns=drop_cols)
y = df[target]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# PREPROCESAMIENTO
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ]
)

# MODELOS

log_reg = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

gboost = Pipeline([
    ("preprocess", preprocess),
    ("model", GradientBoostingClassifier(random_state=42))
])

# ENTRENAR

log_reg.fit(X_train, y_train)
gboost.fit(X_train, y_train)

# EVALUAR (AUC-ROC + ACCURACY)


def eval_model(name, model):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)

    print(f"\n--- {name} ---")
    print(f"AUC-ROC (principal): {auc:.4f}")
    print(f"Accuracy (adicional): {acc:.4f}")
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, pred))
    return auc, acc, proba


auc_log, acc_log, proba_log = eval_model("Logistic Regression", log_reg)
auc_gb,  acc_gb,  proba_gb = eval_model("Gradient Boosting", gboost)

# ELEGIR MEJOR MODELO POR AUC
best_model = gboost if auc_gb >= auc_log else log_reg
print("\nMejor modelo según AUC-ROC:",
      "Gradient Boosting" if best_model == gboost else "Logistic Regression")

# GUARDAR MODELO
joblib.dump(best_model, "modelo_target_EndDate_No.pkl")
print("\nModelo guardado como modelo_target_EndDate_No.pkl")

# FUNCIÓN DE PREDICCIÓN


def predecir_cliente_activo(df_clientes, model):
    """
    Devuelve probabilidad de que EndDate == "No"
    (cliente NO cancelará / seguirá activo)
    """
    proba_activo = model.predict_proba(df_clientes)[:, 1]
    salida = df_clientes.copy()
    salida["prob_EndDate_No"] = proba_activo
    salida["pred_EndDate_No"] = (proba_activo >= 0.5).astype(int)
    return salida.sort_values("prob_EndDate_No", ascending=False)


# EJEMPLO:
predicciones = predecir_cliente_activo(X_test, best_model)
print(predicciones.head(10))
