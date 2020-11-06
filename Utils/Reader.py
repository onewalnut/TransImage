# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:     Reader
   Description :
   Author :       walnut
   date:          2020/10/27
-------------------------------------------------
   Change Activity:
                  2020/10/27:
-------------------------------------------------
"""
__author__ = 'walnut'


import xlrd
import csv
import os


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print
        "---  new folder...  ---"



def read_excel_by_col(file, sheet_index=0):
    workbook = xlrd.open_workbook(r""+file)
    sheet = workbook.sheet_by_index(sheet_index)
    data = []
    for col in range(sheet.ncols):
        data.append(sheet.col_values(col, 0, sheet.nrows))

    return data


def read_excel_by_row(file, sheet_index=0):
    workbook = xlrd.open_workbook(r"" + file)
    sheet = workbook.sheet_by_index(sheet_index)
    data = []
    for row in range(sheet.nrows):
        data.append(sheet.row_values(row, 0, sheet.ncols))

    return data


def read_csv_by_col(file, col_title):
    with open(file, 'r', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        col_values = [row[col_title] for row in reader]
    return col_values


def read_csv_by_row(file, row_index):
    with open(file, 'r', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        row_values = [row for row in reader]
    return row_values[row_index]


def read_csv(file):
    with open(file, 'r', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile)
        row_values = [row for row in reader]
    return row_values
