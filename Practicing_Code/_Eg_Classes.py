# -*- coding: utf-8 -*-
"""
Created on Sat May 16 14:03:35 2020

@author: maria
"""


class Account():
    num = 0
    def __init__(self, money, accountHolder):
        self.accountHolder = accountHolder
        self.balance = money
        Account.num += 1
    
    def deposit(self, amount):
        self.balance += amount
    
    def checkBalance(self):
        print(self.balance)
    
    def withdraw(self, amount):
        self.balance -= amount
    
    
    

drisAccount = Account(10,"Adriana")
lukasAccount = Account(10,"Lukas")


