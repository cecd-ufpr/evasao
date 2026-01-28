# Modelo para Predição de Evasão Universitária
Nesse repositório está disponível o código utilizado para ajuste de um modelo de aprendizado de máquina para prever a probabilidade de evasão de alunos de Ensino Superior. O modelo foi desenvolvido pela Coordenadoria de Estatística e Ciência de Dados da UFPR.

As variáveis utilizadas foram: 
- IRA: medida de desempenho acadêmico que varia entre 0 e 1.
- PropRep: proporção de reprovações, calculada simplesmente pela razão disciplinas em que reprovou / disciplinas cursadas.
- Grau: grau do curso, assumindo valores Bacharelado, Licenciatura, Tecnólogo ou ABI.
- Área do curso: pode ser utilizada tanto Área THE quanto Setor, que no caso da UFPR, é uma unidade administrativa.

O código é fornecido tanto em versão R quanto Python, prioritariamente focando na Regressão Logística, mas sendo também possível o comparativo com outras técnicas de AM.

Também é fornecido um toy example com 30000 observações. A base foi anonimizada para que os alunos não possam ser identificados.

Em caso de dúvidas ou sugestões, entrar em contato através do e-mail cecd.proplad@ufpr.br.

# Model for Predicting University Dropout Rates
This repository contains the code used to adjust a machine learning model to predict the probability of student dropout in higher education. The model was developed by the Statistics and Data Science Coordination of UFPR (Federal University of Paraná).

The variables used were:

- IRA (Academic Performance Index): a measure of academic performance ranging from 0 to 1.
- PropRep (Proportion of Failures): the proportion of failed courses, calculated simply by the ratio of failed courses to courses taken.
- Grau: the degree of the course, assuming values ​​Bachelor's, Licentiate, Technologist, or ABI (Academic, Interdisciplinary, and Complementary).
- Course Area: either THE Area or Sector can be used, which in the case of UFPR is an administrative unit.

The code is provided in both R and Python versions, primarily focusing on Logistic Regression, but also allowing comparison with other ML techniques.

A toy example with 30,000 observations is also provided. The dataset has been anonymized so that students cannot be identified.

For questions or suggestions, please contact us via email at cecd.proplad@ufpr.br.
