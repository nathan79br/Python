#import base de dados
import pandas as pd
import openpyxl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl

tabela_vendas = pd.read_excel('Vendas.xlsx')

#faturamento por loja
faturamento = tabela_vendas[['ID Loja', 'Valor Final']].groupby('ID Loja').sum()

#quantidade de produtos por loja
quantidade_produtos = tabela_vendas[['ID Loja', 'Quantidade']].groupby('ID Loja').sum()

#ticket medio (faturamento / quantidade)
ticket = (faturamento['Valor Final'] / quantidade_produtos['Quantidade']).to_frame('Ticket Médio')

#configura email
sender_email = 'pessoa que enviara o mail'
receiver_email = 'quem recebera o email'
password = 'senha do mail'  # Ou senha de app se estiver usando Gmail com 2FA

message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = 'Relatório de vendas por loja'

body = f''' 
<p>Prezados,</p>

<p>Segue o Relatório de Vendas por cada Loja.</p>
<p>Faturamento:</p>
{faturamento.to_html()}

<p>Quantidade Vendida:</p>
{quantidade_produtos.to_html()}

<p>Ticket Medío dos Produtos em cada Loja:</p>
{ticket.to_html()}

<p>Qualquer Duvida estou a Disposição.</p>
<p>Att.,</p>
<p>Nathan</p>
'''
message.attach(MIMEText(body, "plain"))

# Configurando a conexão com o servidor SMTP
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

print("E-mail enviado com sucesso!")