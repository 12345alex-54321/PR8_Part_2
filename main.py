import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)


def main():



    df = pd.read_csv('bank_scrooge.csv')

    print("\nПервые 10 строк данных:\n", df.head(10))
    print("\nПоследние 10 строк данных:\n", df.tail(10))
    print("\nСлучайные 5 строк (sample):\n", df.sample(5))
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")
    print("\nНазвания столбцов:\n", df.columns.tolist())
    print("\nОбщая информация о датасете:\n", df.info())
    print("\nСтатистическое описание числовых данных:\n", df.describe())
    print("\nСтатистическое описание категориальных данных:\n", df.describe(include = ['str']))




    missing_info = pd.DataFrame({
        'Количество пропусков': df.isnull().sum(),
        'Процент пропусков': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_info = missing_info[missing_info['Количество пропусков'] > 0].sort_values('Количество пропусков', ascending=False)

    if len(missing_info) > 0:
        print("\nСтолбцы с пропущенными значениями:\n", missing_info)
        print(missing_info)
    else:
        print("\nПропущенных значений не обнаружено!")



    print("\nТекущие типы данных в столбцах:\n", df.dtypes)

    type_issues = []

    for col in df.columns:
        if df[col].dtype == 'object':
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            if numeric_values.notna().sum() > len(df) * 0.8:
                type_issues.append(f"{col}: строковый тип, но содержит в основном числа")


    if type_issues:
        print("Обнаружены потенциальные проблемы с типами данных:")
        for issue in type_issues:
            print(f"  - {issue}")
    else:
        print("Явных проблем с типами данных не обнаружено.")



    print("Оригинальные названия столбцов:\n", df.columns.tolist())
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    print("\nИсправленные названия столбцов:\n", df.columns.tolist())


    rows_with_na = df[df.isnull().any(axis=1)]

    if len(rows_with_na) > 0:
        print(f"Количество строк с пропусками: {len(rows_with_na)}")
        print(f"Процент строк с пропусками: {len(rows_with_na) / len(df) * 100:.2f}%")

        print("\nПримеры строк с пропусками:\n", rows_with_na.head())

        print("\nАнализ связи пропусков с целевой переменной:\n")
        if 'churn' in df.columns:
            churn_rate_with_na = rows_with_na['churn'].mean() if len(rows_with_na) > 0 else 0
            churn_rate_without_na = df.dropna()['churn'].mean() if len(df.dropna()) > 0 else 0
            print(f"Churn rate для строк с пропусками: {churn_rate_with_na:.2%}")
            print(f"Churn rate для строк без пропусков: {churn_rate_without_na:.2%}")
    else:
        print("Строк с пропусками не найдено.")



    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df.fillna({col: median_value}, inplace=True)
            print(f"  - Столбец '{col}': пропуски заполнены медианой ({median_value:.2f})")

    categorical_columns = df.select_dtypes(include=['str']).columns
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col].fillna(mode_value, inplace=True)
            print(f"  - Столбец '{col}': пропуски заполнены модой ('{mode_value}')")

    remaining_na = df.isnull().sum().sum()
    if remaining_na > 0:
        print(f"\nУдаляем оставшиеся строки с пропусками ({remaining_na} значений)")
        df.dropna(inplace=True)

    print(f"\nПосле обработки пропусков:")
    print(f"Количество строк: {len(df)}")
    print(f"Пропущенных значений осталось: {df.isnull().sum().sum()}")


    full_duplicates = df.duplicated().sum()
    print(f"Полных дубликатов строк: {full_duplicates}")

    if full_duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"Удалено {full_duplicates} полных дубликатов")

    if 'user_id' in df.columns:
        duplicate_users = df['user_id'].duplicated().sum()
        print(f"Дубликатов по user_id: {duplicate_users}")

        if duplicate_users > 0:

            print("\nПримеры дублирующихся user_id:")
            duplicate_user_ids = df[df['user_id'].duplicated(keep=False)]['user_id'].unique()[:5]

            for uid in duplicate_user_ids:
                print(f"\nUser ID: {uid}")
                print(df[df['user_id'] == uid])

            df.drop_duplicates(subset=['user_id'], keep='last', inplace=True)
            print(f"После удаления дубликатов: {len(df)} строк")



    categorical_columns = df.select_dtypes(include=['str']).columns

    for col in categorical_columns:
        print(f"\nАнализ столбца '{col}':")
        unique_values = df[col].unique()
        print(f"  Уникальных значений: {len(unique_values)}")

        value_counts = df[col].value_counts()
        print(f"  Распределение значений (первые 10):")
        for val, count in value_counts.head(10).items():
            print(f"    '{val}': {count}")

        if df[col].dtype == 'object':

            df[col] = df[col].str.strip()

        if 'gender' in col.lower() or col.lower() == 'sex':
            print("\n  Стандартизация значений gender:")
            gender_mapping = {
                'M': 'Male', 'm': 'Male', 'male': 'Male', 'MALE': 'Male',
                'F': 'Female', 'f': 'Female', 'female': 'Female', 'FEMALE': 'Female',
                '0': 'Female', '1': 'Male'
            }
            df[col] = df[col].replace(gender_mapping)
            print(f"  Уникальные значения после стандартизации: {df[col].unique()}")


    df_encoded = df.copy()

    categorical_columns = df_encoded.select_dtypes(include=['str']).columns.tolist()

    if 'user_id' in categorical_columns:
        categorical_columns.remove('user_id')

    print(f"\nКатегориальные столбцы для кодирования: {categorical_columns}")

    for col in categorical_columns:
        n_unique = df_encoded[col].nunique()
        print(f"\nКодирование столбца '{col}' ({n_unique} уникальных значений)")

        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)

    print(df_encoded.head())

if __name__ == "__main__":
    main()