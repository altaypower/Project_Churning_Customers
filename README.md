# Project_Churning_Customers

Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy API: flask Данные: с https://leaps.analyttica.com/home

Задача: предсказать по описанию клиента банка вероятность его ухода. Бинарная классификация

Используемые признаки:

Total_Trans_Amt          
Total_Trans_Ct            
Total_Relationship_Count  
Total_Amt_Chng_Q4_Q1      
Months_Inactive_12_mon    
Total_Revolving_Bal 
Внутри пайплайна генерируется еще 4 признака.

Модель: CatBoostClassifier
Клонируем репозиторий и создаем образ

$ git clone git@github.com:altaypower/GB_Project_Churning_Customers.git
$ cd GB_Project_Churning_Customers
$ docker build -t altaypower/GB_Project_Churning_Customers .

Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)

$ docker run -d -p 8180:8180 -p -v <your_local_path_to_pretrained_models>:/app/app/models altaypower/GB_Project_Churning_Customers

Также приложен ноутбук с базовым решением.