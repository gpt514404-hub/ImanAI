"""
🧠 ImanCore v3.0 - هسته مرکزی هوش مصنوعی کامل
✅ مدیریت کامل API Keyها
✅ پردازش هوشمند فایل‌ها
✅ یادگیری Real-Time + دیپ لرنینگ داخلی
✅ پیش‌بینی‌های هوشمند
✅ رابط کاربری گرافیکی + API سرور RESTful
✅ سازگار با 32-bit
✅ امنیت کامل
✅ دیپ لرنینگ داخلی (بدون وابستگی خارجی)
✅ پایگاه داده SQLite
✅ سیستم Real-Time
"""

import sys
import os
import json
import sqlite3
import threading
import hashlib
import uuid
import datetime
import time
import secrets
import base64
import io
import mimetypes
import zipfile
import csv
import re
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import queue

# ============================================================================
# PyQt5 برای GUI
# ============================================================================
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTabWidget, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
        QGroupBox, QTextEdit, QLineEdit, QFormLayout, QDialog, QDialogButtonBox,
        QFileDialog, QMessageBox, QStatusBar, QCheckBox, QSpinBox,
        QDoubleSpinBox, QComboBox, QProgressDialog, QGridLayout,
        QAbstractItemView, QHeaderView, QListWidget, QListWidgetItem,
        QProgressBar, QToolBar, QMenu, QMenuBar, QAction, QSplitter,
        QFrame, QTextBrowser, QScrollArea, QDateEdit, QTimeEdit
    )
    from PyQt5.QtCore import (
        Qt, QTimer, QDateTime, QDate, QTime, QSize, 
        QThread, pyqtSignal, QObject, QRunnable, QThreadPool,
        QPoint, QRect, QEvent
    )
    from PyQt5.QtGui import (
        QIcon, QPixmap, QColor, QFont, QFontDatabase, 
        QPainter, QPen, QBrush, QPalette, QKeySequence,
        QDesktopServices, QMovie, QCursor,
        QPaintEvent, QResizeEvent
    )
    PYQT_AVAILABLE = True
except ImportError as e:
    PYQT_AVAILABLE = False
    print(f"⚠️ PyQt5 نصب نیست یا خطا در بارگذاری: {e}")
    print("📦 نصب: pip install PyQt5")

# ============================================================================
# FastAPI برای API سرور
# ============================================================================
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form, Query, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field, validator
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError as e:
    FASTAPI_AVAILABLE = False
    print(f"⚠️ FastAPI نصب نیست یا خطا در بارگذاری: {e}")
    print("📦 نصب: pip install fastapi uvicorn")

# ============================================================================
# کتابخانه‌های پردازش فایل
# ============================================================================
try:
    from PIL import Image
    # خواندن PDF با کتابخانه‌های مختلف
    try:
        import fitz  # PyMuPDF
        PDF_AVAILABLE = True
        PDF_LIB = "pymupdf"
    except ImportError:
        try:
            from pdfminer.high_level import extract_text as pdf_extract_text
            PDF_AVAILABLE = True
            PDF_LIB = "pdfminer"
        except ImportError:
            PDF_AVAILABLE = False
            PDF_LIB = None
    
    import pandas as pd
    import numpy as np
    PROCESSING_AVAILABLE = True
except ImportError as e:
    PROCESSING_AVAILABLE = False
    PDF_AVAILABLE = False
    print(f"⚠️ برخی کتابخانه‌های پردازش فایل نصب نیستند: {e}")

# ============================================================================
# دیپ لرنینگ داخلی (کاملاً مستقل)
# ============================================================================

class NeuralNetwork:
    """شبکه عصبی کاملاً داخلی - بدون وابستگی خارجی"""
    
    def __init__(self, layers: List[int], 
                 activation: str = 'relu',
                 output_activation: str = 'softmax',
                 learning_rate: float = 0.01):
        self.layers = layers
        self.activation = activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        
        self.weights = []
        self.biases = []
        self.history = {'train_loss': [], 'train_accuracy': [], 
                       'val_loss': [], 'val_accuracy': []}
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """مقداردهی اولیه وزن‌ها و بایاس‌ها"""
        random.seed(42)
        
        for i in range(len(self.layers) - 1):
            scale = math.sqrt(2.0 / self.layers[i])
            
            W = []
            for _ in range(self.layers[i]):
                row = []
                for _ in range(self.layers[i + 1]):
                    row.append(random.gauss(0, scale))
                W.append(row)
            
            b = [[0.0] * self.layers[i + 1]]
            
            self.weights.append(W)
            self.biases.append(b)
    
    def _sigmoid(self, x: float) -> float:
        if x > 100: return 1.0
        elif x < -100: return 0.0
        return 1.0 / (1.0 + math.exp(-x))
    
    def _relu(self, x: float) -> float:
        return max(0, x)
    
    def _leaky_relu(self, x: float) -> float:
        return x if x > 0 else 0.01 * x
    
    def _tanh(self, x: float) -> float:
        return math.tanh(x)
    
    def _softmax(self, x: List[float]) -> List[float]:
        max_val = max(x)
        exp_values = [math.exp(val - max_val) for val in x]
        sum_exp = sum(exp_values)
        return [val / sum_exp for val in exp_values]
    
    def _activation_function(self, x: float, derivative: bool = False) -> float:
        if self.activation == 'relu':
            if derivative:
                return 1.0 if x > 0 else 0.0
            return self._relu(x)
        elif self.activation == 'leaky_relu':
            if derivative:
                return 1.0 if x > 0 else 0.01
            return self._leaky_relu(x)
        elif self.activation == 'sigmoid':
            s = self._sigmoid(x)
            if derivative:
                return s * (1 - s)
            return s
        elif self.activation == 'tanh':
            t = self._tanh(x)
            if derivative:
                return 1 - t * t
            return t
        else:
            return x
    
    def _matrix_multiply(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                sum_val = 0.0
                for k in range(len(A[0])):
                    sum_val += A[i][k] * B[k][j]
                row.append(sum_val)
            result.append(row)
        return result
    
    def _matrix_add(self, A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(A[i][j] + B[i][j])
            result.append(row)
        return result
    
    def _transpose(self, A: List[List[float]]) -> List[List[float]]:
        result = []
        for j in range(len(A[0])):
            row = []
            for i in range(len(A)):
                row.append(A[i][j])
            result.append(row)
        return result
    
    def forward(self, X: List[List[float]]) -> List[List[float]]:
        self.layer_outputs = []
        self.activated_outputs = []
        
        current_input = X
        self.layer_outputs.append(current_input)
        
        for i in range(len(self.weights) - 1):
            Z = self._matrix_multiply(current_input, self.weights[i])
            Z = self._matrix_add(Z, self.biases[i])
            self.layer_outputs.append(Z)
            
            A = []
            for row in Z:
                activated_row = [self._activation_function(val) for val in row]
                A.append(activated_row)
            
            self.activated_outputs.append(A)
            current_input = A
        
        Z_out = self._matrix_multiply(current_input, self.weights[-1])
        Z_out = self._matrix_add(Z_out, self.biases[-1])
        self.layer_outputs.append(Z_out)
        
        if self.output_activation == 'softmax':
            A_out = []
            for row in Z_out:
                A_out.append(self._softmax(row))
        elif self.output_activation == 'sigmoid':
            A_out = []
            for row in Z_out:
                A_out.append([self._sigmoid(val) for val in row])
        else:
            A_out = Z_out
        
        self.activated_outputs.append(A_out)
        return A_out
    
    def compute_loss(self, y_pred: List[List[float]], y_true: List[List[float]]) -> float:
        m = len(y_true)
        loss = 0.0
        
        if self.output_activation == 'softmax':
            for i in range(m):
                for j in range(len(y_true[i])):
                    pred = max(min(y_pred[i][j], 1 - 1e-8), 1e-8)
                    loss -= y_true[i][j] * math.log(pred)
        
        elif self.output_activation == 'sigmoid':
            for i in range(m):
                pred = max(min(y_pred[i][0], 1 - 1e-8), 1e-8)
                loss -= y_true[i][0] * math.log(pred) + (1 - y_true[i][0]) * math.log(1 - pred)
        
        else:
            for i in range(m):
                for j in range(len(y_true[i])):
                    loss += (y_pred[i][j] - y_true[i][j]) ** 2
        
        loss /= m
        
        lambda_reg = 0.001
        reg_loss = 0.0
        for W in self.weights:
            for row in W:
                for val in row:
                    reg_loss += val * val
        
        loss += (lambda_reg / (2 * m)) * reg_loss
        
        return loss
    
    def backward(self, X: List[List[float]], y_true: List[List[float]]):
        m = len(X)
        L = len(self.weights)
        
        dW = [None] * L
        db = [None] * L
        
        if self.output_activation == 'softmax':
            dZ = []
            for i in range(m):
                row = []
                for j in range(len(y_true[i])):
                    row.append(self.activated_outputs[-1][i][j] - y_true[i][j])
                dZ.append(row)
        else:
            dZ = []
            for i in range(m):
                row = []
                for j in range(len(y_true[i])):
                    row.append(self.activated_outputs[-1][i][j] - y_true[i][j])
                dZ.append(row)
        
        A_prev = self.activated_outputs[-2] if L > 1 else X
        dW[-1] = self._matrix_multiply(self._transpose(A_prev), dZ)
        
        for i in range(len(dW[-1])):
            for j in range(len(dW[-1][0])):
                dW[-1][i][j] /= m
        
        db[-1] = [[sum(dZ[i][j] for i in range(m)) / m for j in range(len(dZ[0]))]]
        
        for l in range(L - 2, -1, -1):
            dA = self._matrix_multiply(dZ, self._transpose(self.weights[l + 1]))
            
            Z = self.layer_outputs[l + 1]
            dZ = []
            for i in range(m):
                row = []
                for j in range(len(dA[0])):
                    grad = dA[i][j] * self._activation_function(Z[i][j], derivative=True)
                    row.append(grad)
                dZ.append(row)
            
            A_prev = self.activated_outputs[l - 1] if l > 0 else X
            
            dW[l] = self._matrix_multiply(self._transpose(A_prev), dZ)
            for i in range(len(dW[l])):
                for j in range(len(dW[l][0])):
                    dW[l][i][j] /= m
            
            db[l] = [[sum(dZ[i][j] for i in range(m)) / m for j in range(len(dZ[0]))]]
        
        self.dW = dW
        self.db = db
    
    def update_parameters(self):
        for l in range(len(self.weights)):
            for i in range(len(self.weights[l])):
                for j in range(len(self.weights[l][0])):
                    self.weights[l][i][j] -= self.learning_rate * self.dW[l][i][j]
            
            for j in range(len(self.biases[l][0])):
                self.biases[l][0][j] -= self.learning_rate * self.db[l][0][j]
    
    def train(self, X: List[List[float]], y: List[List[float]], 
              epochs: int = 100, batch_size: int = 32,
              validation_data: tuple = None) -> Dict[str, List]:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        m = len(X)
        
        for epoch in range(epochs):
            indices = list(range(m))
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            y_shuffled = [y[i] for i in indices]
            
            epoch_loss = 0.0
            correct = 0
            
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]
                
                y_pred = self.forward(X_batch)
                batch_loss = self.compute_loss(y_pred, y_batch)
                epoch_loss += batch_loss
                
                if self.output_activation == 'softmax':
                    for j in range(len(y_batch)):
                        pred_idx = max(range(len(y_pred[j])), key=lambda k: y_pred[j][k])
                        true_idx = max(range(len(y_batch[j])), key=lambda k: y_batch[j][k])
                        if pred_idx == true_idx:
                            correct += 1
                elif self.output_activation == 'sigmoid':
                    for j in range(len(y_batch)):
                        pred_class = 1 if y_pred[j][0] > 0.5 else 0
                        if pred_class == int(y_batch[j][0] + 0.5):
                            correct += 1
                
                self.backward(X_batch, y_batch)
                self.update_parameters()
            
            avg_loss = epoch_loss / (m / batch_size)
            accuracy = correct / m
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if validation_data:
                X_val, y_val = validation_data
                y_pred_val = self.forward(X_val)
                val_loss = self.compute_loss(y_pred_val, y_val)
                val_losses.append(val_loss)
                
                val_correct = 0
                if self.output_activation == 'softmax':
                    for j in range(len(y_val)):
                        pred_idx = max(range(len(y_pred_val[j])), key=lambda k: y_pred_val[j][k])
                        true_idx = max(range(len(y_val[j])), key=lambda k: y_val[j][k])
                        if pred_idx == true_idx:
                            val_correct += 1
                val_accuracies.append(val_correct / len(y_val))
            
            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
                if validation_data:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)
        
        self.history = {
            'train_loss': train_losses,
            'train_accuracy': train_accuracies,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies
        }
        
        return self.history
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        return self.forward(X)
    
    def predict_classes(self, X: List[List[float]]) -> List[int]:
        y_pred = self.predict(X)
        
        if self.output_activation == 'softmax':
            return [max(range(len(pred)), key=lambda i: pred[i]) for pred in y_pred]
        elif self.output_activation == 'sigmoid':
            return [1 if pred[0] > 0.5 else 0 for pred in y_pred]
        else:
            return [int(pred[0] + 0.5) for pred in y_pred]
    
    def save(self, filepath: str):
        model_data = {
            'layers': self.layers,
            'activation': self.activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'weights': self.weights,
            'biases': self.biases,
            'history': self.history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.layers = model_data['layers']
        self.activation = model_data['activation']
        self.output_activation = model_data['output_activation']
        self.learning_rate = model_data['learning_rate']
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.history = model_data['history']
    
    def summary(self):
        print("=" * 60)
        print("🧠 Neural Network Summary")
        print("=" * 60)
        print(f"Architecture: {self.layers}")
        print(f"Hidden Activation: {self.activation}")
        print(f"Output Activation: {self.output_activation}")
        print(f"Learning Rate: {self.learning_rate}")
        
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            layer_params = len(W) * len(W[0]) + len(b[0])
            total_params += layer_params
            print(f"Layer {i+1}: {len(W[0])} neurons, {layer_params:,} parameters")
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Training History: {len(self.history['train_loss'])} epochs")
        
        if self.history['train_loss']:
            print(f"Final Train Loss: {self.history['train_loss'][-1]:.4f}")
            print(f"Final Train Accuracy: {self.history['train_accuracy'][-1]:.4f}")
            if self.history['val_loss']:
                print(f"Final Val Loss: {self.history['val_loss'][-1]:.4f}")
                print(f"Final Val Accuracy: {self.history['val_accuracy'][-1]:.4f}")
        
        print("=" * 60)

# ============================================================================
# ساختارهای داده
# ============================================================================

class FileType(Enum):
    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    ZIP = "zip"
    UNKNOWN = "unknown"

@dataclass
class APIKeyInfo:
    key_id: str
    app_name: str
    owner: str
    permissions: List[str]
    rate_limit: int = 1000
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    is_active: bool = True
    total_requests: int = 0
    
    def to_dict(self):
        return {
            'key_id': self.key_id,
            'app_name': self.app_name,
            'owner': self.owner,
            'permissions': self.permissions,
            'rate_limit': self.rate_limit,
            'created_at': self.created_at,
            'last_used': self.last_used,
            'is_active': self.is_active,
            'total_requests': self.total_requests
        }

@dataclass
class FileRecord:
    file_id: str
    original_name: str
    file_type: FileType
    size_bytes: int
    hash_md5: str
    uploaded_by: str
    upload_time: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    storage_path: str = ""
    
    def to_dict(self):
        return {
            'file_id': self.file_id,
            'original_name': self.original_name,
            'file_type': self.file_type.value,
            'size_bytes': self.size_bytes,
            'hash_md5': self.hash_md5,
            'uploaded_by': self.uploaded_by,
            'upload_time': self.upload_time,
            'processed': self.processed,
            'metadata': self.metadata,
            'storage_path': self.storage_path
        }

# ============================================================================
# سیستم مدیریت API Key
# ============================================================================

class APIKeyManager:
    def __init__(self, db_path: str = "imancore_keys.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                api_key TEXT UNIQUE NOT NULL,
                app_name TEXT NOT NULL,
                owner TEXT NOT NULL,
                permissions TEXT NOT NULL,
                rate_limit INTEGER DEFAULT 1000,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                total_requests INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                key_id TEXT NOT NULL,
                hour TIMESTAMP NOT NULL,
                request_count INTEGER DEFAULT 0,
                PRIMARY KEY (key_id, hour)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_api_key(self, app_name: str, owner: str, 
                        permissions: List[str] = None) -> Tuple[str, str]:
        if permissions is None:
            permissions = ['upload', 'learn', 'predict', 'query']
        
        key_id = str(uuid.uuid4())
        api_key = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_keys 
            (key_id, api_key, app_name, owner, permissions, rate_limit)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            key_id,
            api_key,
            app_name,
            owner,
            json.dumps(permissions),
            1000
        ))
        
        conn.commit()
        conn.close()
        
        return key_id, api_key
    
    def validate_api_key(self, api_key: str, permission: str = None) -> Optional[APIKeyInfo]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM api_keys 
            WHERE api_key = ? AND is_active = TRUE
        ''', (api_key,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        
        current_hour = datetime.datetime.now().strftime('%Y-%m-%d %H:00:00')
        cursor.execute('''
            SELECT request_count FROM rate_limits 
            WHERE key_id = ? AND hour = ?
        ''', (row['key_id'], current_hour))
        
        rate_row = cursor.fetchone()
        current_count = rate_row['request_count'] if rate_row else 0
        
        if current_count >= row['rate_limit']:
            conn.close()
            return None
        
        if rate_row:
            cursor.execute('''
                UPDATE rate_limits 
                SET request_count = request_count + 1 
                WHERE key_id = ? AND hour = ?
            ''', (row['key_id'], current_hour))
        else:
            cursor.execute('''
                INSERT INTO rate_limits (key_id, hour, request_count)
                VALUES (?, ?, 1)
            ''', (row['key_id'], current_hour))
        
        cursor.execute('''
            UPDATE api_keys 
            SET last_used = ?, total_requests = total_requests + 1 
            WHERE key_id = ?
        ''', (datetime.datetime.now().isoformat(), row['key_id']))
        
        conn.commit()
        
        if permission:
            permissions = json.loads(row['permissions'])
            if permission not in permissions:
                conn.close()
                return None
        
        key_info = APIKeyInfo(
            key_id=row['key_id'],
            app_name=row['app_name'],
            owner=row['owner'],
            permissions=json.loads(row['permissions']),
            rate_limit=row['rate_limit'],
            created_at=row['created_at'],
            last_used=row['last_used'],
            is_active=bool(row['is_active']),
            total_requests=row['total_requests']
        )
        
        conn.close()
        return key_info
    
    def list_api_keys(self) -> List[APIKeyInfo]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM api_keys ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        keys = []
        for row in rows:
            keys.append(APIKeyInfo(
                key_id=row['key_id'],
                app_name=row['app_name'],
                owner=row['owner'],
                permissions=json.loads(row['permissions']),
                rate_limit=row['rate_limit'],
                created_at=row['created_at'],
                last_used=row['last_used'],
                is_active=bool(row['is_active']),
                total_requests=row['total_requests']
            ))
        
        return keys
    
    def revoke_api_key(self, api_key: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE api_keys 
            SET is_active = FALSE 
            WHERE api_key = ?
        ''', (api_key,))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def delete_api_key(self, key_id: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM api_keys WHERE key_id = ?', (key_id,))
        cursor.execute('DELETE FROM rate_limits WHERE key_id = ?', (key_id,))
        
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def update_api_key(self, key_id: str, **kwargs) -> bool:
        allowed_fields = ['app_name', 'owner', 'permissions', 'rate_limit', 'is_active']
        
        updates = []
        params = []
        
        for field, value in kwargs.items():
            if field in allowed_fields:
                if field == 'permissions':
                    updates.append(f"{field} = ?")
                    params.append(json.dumps(value))
                else:
                    updates.append(f"{field} = ?")
                    params.append(value)
        
        if not updates:
            return False
        
        params.append(key_id)
        query = f"UPDATE api_keys SET {', '.join(updates)} WHERE key_id = ?"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(query, params)
        affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def get_key_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM api_keys')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM api_keys WHERE is_active = TRUE')
        active = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(total_requests) FROM api_keys')
        total_requests = cursor.fetchone()[0] or 0
        
        cursor.execute('''
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM api_keys 
            GROUP BY DATE(created_at) 
            ORDER BY date DESC 
            LIMIT 7
        ''')
        
        daily_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_keys': total,
            'active_keys': active,
            'inactive_keys': total - active,
            'total_requests': total_requests,
            'daily_stats': [{'date': row[0], 'count': row[1]} for row in daily_stats]
        }

# ============================================================================
# هسته پردازش فایل
# ============================================================================

class FileProcessor:
    def __init__(self, storage_path: str = "./imancore_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        folders = ["images", "pdfs", "texts", "data", "models", "exports", "temp"]
        for folder in folders:
            (self.storage_path / folder).mkdir(exist_ok=True)
    
    def detect_file_type(self, filename: str) -> FileType:
        ext = Path(filename).suffix.lower()
        
        ext_to_type = {
            '.txt': FileType.TEXT,
            '.csv': FileType.CSV,
            '.json': FileType.JSON,
            '.pdf': FileType.PDF,
            '.jpg': FileType.IMAGE,
            '.jpeg': FileType.IMAGE,
            '.png': FileType.IMAGE,
            '.gif': FileType.IMAGE,
            '.bmp': FileType.IMAGE,
            '.xlsx': FileType.EXCEL,
            '.xls': FileType.EXCEL,
            '.zip': FileType.ZIP,
            '.tar': FileType.ZIP,
            '.gz': FileType.ZIP,
            '.rar': FileType.ZIP,
            '.xml': FileType.TEXT,
            '.html': FileType.TEXT,
            '.htm': FileType.TEXT
        }
        
        return ext_to_type.get(ext, FileType.UNKNOWN)
    
    def calculate_hash(self, content: bytes) -> str:
        return hashlib.md5(content).hexdigest()
    
    def save_file(self, file_id: str, filename: str, content: bytes) -> Tuple[str, FileType]:
        file_type = self.detect_file_type(filename)
        hash_md5 = self.calculate_hash(content)
        
        folder_map = {
            FileType.IMAGE: "images",
            FileType.PDF: "pdfs",
            FileType.TEXT: "texts",
            FileType.CSV: "data",
            FileType.JSON: "data",
            FileType.EXCEL: "data",
            FileType.ZIP: "data",
            FileType.UNKNOWN: "data"
        }
        
        folder = folder_map.get(file_type, "data")
        
        safe_name = re.sub(r'[^\w\-_.]', '_', filename)
        unique_name = f"{file_id}_{hash_md5[:8]}_{safe_name}"
        save_path = self.storage_path / folder / unique_name
        
        with open(save_path, 'wb') as f:
            f.write(content)
        
        return str(save_path), file_type
    
    def extract_text_from_file(self, file_path: str, file_type: FileType) -> str:
        try:
            if file_type == FileType.TEXT:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_type == FileType.PDF and PDF_AVAILABLE:
                if PDF_LIB == "pymupdf":
                    with fitz.open(file_path) as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text() + "\n"
                        return text
                elif PDF_LIB == "pdfminer":
                    return pdf_extract_text(file_path)
                else:
                    return f"[PDF file - no reader available]"
            
            elif file_type == FileType.CSV:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_type == FileType.JSON:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            else:
                return f"[Binary file of type: {file_type.value}]"
                
        except Exception as e:
            return f"[Error extracting text: {str(e)}]"
    
    def extract_metadata(self, file_path: str, file_type: FileType) -> Dict[str, Any]:
        metadata = {
            'file_type': file_type.value,
            'size_bytes': os.path.getsize(file_path),
            'last_modified': datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat()
        }
        
        try:
            if file_type == FileType.IMAGE and PROCESSING_AVAILABLE:
                with Image.open(file_path) as img:
                    metadata.update({
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format
                    })
            
            elif file_type == FileType.PDF and PDF_AVAILABLE:
                if PDF_LIB == "pymupdf":
                    with fitz.open(file_path) as doc:
                        metadata['page_count'] = len(doc)
                elif PDF_LIB == "pdfminer":
                    metadata['page_count'] = 'unknown'
            
            elif file_type == FileType.TEXT:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    metadata.update({
                        'text_length': len(text),
                        'line_count': text.count('\n') + 1,
                        'word_count': len(text.split())
                    })
            
            elif file_type == FileType.CSV:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    if lines:
                        metadata.update({
                            'row_count': len(lines) - 1,
                            'header': lines[0].strip().split(',')
                        })
        
        except Exception as e:
            metadata['error'] = str(e)
        
        return metadata
    
    def compress_files(self, file_paths: List[str], output_path: str) -> bool:
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        arcname = os.path.basename(file_path)
                        zipf.write(file_path, arcname)
            return True
        except Exception as e:
            print(f"Error compressing files: {e}")
            return False

# ============================================================================
# هسته یادگیری Real-Time
# ============================================================================

class LearningEngine:
    def __init__(self, storage_path: str = "./imancore_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.models = {}
        self.deep_models = {}
        self.learning_queue = queue.Queue()
        self.is_running = True
        
        self.load_models()
        self._start_processing_thread()
        
        print("🧠 Learning Engine initialized")
    
    def _start_processing_thread(self):
        def processor():
            while self.is_running:
                try:
                    record_data = self.learning_queue.get(timeout=1)
                    if record_data:
                        self._process_learning_record(record_data)
                    self.learning_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error in learning processor: {e}")
        
        thread = threading.Thread(target=processor, daemon=True)
        thread.start()
    
    def add_learning_record(self, source_app: str, data_type: str, 
                           content: str, labels: List[str] = None):
        if labels is None:
            labels = []
        
        record_data = {
            'record_id': str(uuid.uuid4()),
            'source_app': source_app,
            'data_type': data_type,
            'content': content,
            'labels': labels,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.learning_queue.put(record_data)
    
    def _process_learning_record(self, record_data: Dict):
        try:
            record_id = record_data['record_id']
            source_app = record_data['source_app']
            data_type = record_data['data_type']
            content = record_data['content']
            labels = record_data['labels']
            
            features = self._extract_features(content, data_type)
            
            use_deep_learning = self._should_use_deep_learning(content, data_type, labels)
            
            if use_deep_learning:
                model_key = f"{source_app}_{data_type}_deep"
                self._train_deep_model(model_key, features, labels)
            else:
                model_key = f"{source_app}_{data_type}"
                self._update_traditional_model(model_key, content, labels, features)
            
            print(f"✅ Learning processed for {source_app}: {data_type}")
            
        except Exception as e:
            print(f"❌ Error processing learning record: {e}")
    
    def _extract_features(self, content: str, data_type: str) -> Dict[str, Any]:
        features = {}
        
        if data_type == 'text':
            features['length'] = len(content)
            features['word_count'] = len(content.split())
            features['char_count'] = len(content)
            features['has_numbers'] = bool(re.search(r'\d+', content))
            features['has_special_chars'] = bool(re.search(r'[^\w\s]', content))
            features['avg_word_length'] = sum(len(word) for word in content.split()) / max(len(content.split()), 1)
        
        elif data_type == 'numeric':
            try:
                numbers = [float(x) for x in content.split() if self._is_number(x)]
                if numbers:
                    features['count'] = len(numbers)
                    features['mean'] = sum(numbers) / len(numbers)
                    features['min'] = min(numbers)
                    features['max'] = max(numbers)
                    if len(numbers) > 1:
                        variance = sum((x - features['mean']) ** 2 for x in numbers) / (len(numbers) - 1)
                        features['std'] = math.sqrt(variance)
                    else:
                        features['std'] = 0
            except:
                features['error'] = 'Invalid numeric data'
        
        elif data_type == 'categorical':
            items = content.split(',')
            features['unique_count'] = len(set(items))
            features['total_count'] = len(items)
            if items:
                counts = {}
                for item in items:
                    counts[item] = counts.get(item, 0) + 1
                most_common = max(counts.items(), key=lambda x: x[1])
                features['most_common'] = most_common[0]
                features['most_common_count'] = most_common[1]
        
        return features
    
    def _is_number(self, s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _should_use_deep_learning(self, content: str, data_type: str, labels: List[str]) -> bool:
        criteria = [
            len(content) > 1000,
            len(labels) > 3,
            data_type in ['image', 'complex_text', 'time_series'],
            'deep_learning' in labels,
            data_type == 'text' and len(content.split()) > 100
        ]
        
        return any(criteria)
    
    def _train_deep_model(self, model_key: str, features: Dict, labels: List[str]):
        if model_key not in self.deep_models:
            input_size = 10
            output_size = len(set(labels)) if labels and len(set(labels)) > 1 else 2
            
            self.deep_models[model_key] = NeuralNetwork(
                layers=[input_size, 16, 8, output_size],
                activation='relu',
                output_activation='softmax',
                learning_rate=0.01
            )
            
            print(f"🧠 Created new deep learning model: {model_key}")
        
        X = self._prepare_features_for_training(features)
        y = self._prepare_labels_for_training(labels)
        
        if X and y:
            try:
                self.deep_models[model_key].train(X, y, epochs=10, batch_size=4)
                print(f"✅ Deep model trained: {model_key}")
                self.save_models()
            except Exception as e:
                print(f"❌ Error training deep model: {e}")
    
    def _prepare_features_for_training(self, features: Dict) -> List[List[float]]:
        feature_vector = []
        
        numeric_features = ['length', 'word_count', 'char_count', 'avg_word_length', 
                          'count', 'mean', 'min', 'max', 'std', 'unique_count']
        
        for feat in numeric_features:
            if feat in features:
                value = float(features[feat])
                if feat == 'length':
                    value = min(value / 1000, 1.0)
                elif feat == 'word_count':
                    value = min(value / 100, 1.0)
                feature_vector.append(value)
            else:
                feature_vector.append(0.0)
        
        while len(feature_vector) < 10:
            feature_vector.append(0.0)
        
        return [feature_vector[:10]]
    
    def _prepare_labels_for_training(self, labels: List[str]) -> List[List[float]]:
        if not labels:
            return [[1.0, 0.0]]
        
        unique_labels = list(set(labels))
        if len(unique_labels) == 1:
            return [[1.0, 0.0]]
        else:
            label_index = unique_labels.index(labels[0])
            one_hot = [0.0] * len(unique_labels)
            one_hot[label_index] = 1.0
            return [one_hot]
    
    def _update_traditional_model(self, model_key: str, content: str, 
                                 labels: List[str], features: Dict):
        if model_key not in self.models:
            self.models[model_key] = {
                'type': 'traditional',
                'samples': [],
                'patterns': {},
                'label_stats': {},
                'last_updated': datetime.datetime.now().isoformat()
            }
        
        model = self.models[model_key]
        
        model['samples'].append({
            'content': content[:200],
            'features': features,
            'labels': labels,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        if len(model['samples']) > 1000:
            model['samples'] = model['samples'][-1000:]
        
        for label in labels:
            if label not in model['label_stats']:
                model['label_stats'][label] = {
                    'count': 1,
                    'first_seen': datetime.datetime.now().isoformat(),
                    'last_seen': datetime.datetime.now().isoformat()
                }
            else:
                model['label_stats'][label]['count'] += 1
                model['label_stats'][label]['last_seen'] = datetime.datetime.now().isoformat()
        
        model['last_updated'] = datetime.datetime.now().isoformat()
        
        print(f"📊 Traditional model updated: {model_key} ({len(model['samples'])} samples)")
    
    def predict(self, model_key: str, data_type: str, content: str) -> Dict[str, Any]:
        result = {
            'predictions': [],
            'confidence': 0.0,
            'model_type': 'unknown',
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        deep_key = f"{model_key}_{data_type}_deep"
        if deep_key in self.deep_models:
            try:
                model = self.deep_models[deep_key]
                features = self._extract_features(content, data_type)
                X = self._prepare_features_for_training(features)
                
                predictions = model.predict(X)
                result['model_type'] = 'deep_learning'
                result['predictions'] = predictions[0] if predictions else []
                result['confidence'] = max(predictions[0]) if predictions else 0.0
                
                return result
            except Exception as e:
                print(f"Deep learning prediction failed: {e}")
        
        key = f"{model_key}_{data_type}"
        if key in self.models:
            model = self.models[key]
            result['model_type'] = 'traditional'
            
            content_lower = content.lower()
            predictions = []
            
            for label, stats in model.get('label_stats', {}).items():
                confidence = min(0.9, stats['count'] / 100)
                
                if label.lower() in content_lower:
                    confidence = max(confidence, 0.7)
                
                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'support': stats['count'],
                    'last_seen': stats['last_seen']
                })
            
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            result['predictions'] = predictions[:5]
            
            if predictions:
                result['confidence'] = predictions[0]['confidence']
        
        return result
    
    def train_deep_model_custom(self, model_key: str, X: List[List[float]], 
                               y: List[List[float]], epochs: int = 100,
                               layers: List[int] = None) -> Dict[str, Any]:
        try:
            if layers is None:
                layers = [len(X[0]), 32, 16, len(y[0])]
            
            model = NeuralNetwork(
                layers=layers,
                activation='relu',
                output_activation='softmax',
                learning_rate=0.01
            )
            
            history = model.train(X, y, epochs=epochs, batch_size=32)
            
            self.deep_models[model_key] = model
            self.save_models()
            
            return {
                'success': True,
                'model_key': model_key,
                'final_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                'final_accuracy': history['train_accuracy'][-1] if history['train_accuracy'] else 0,
                'layers': layers
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_models(self):
        try:
            for model_key, model in self.deep_models.items():
                if hasattr(model, 'save'):
                    safe_key = re.sub(r'[^\w\-_]', '_', model_key)
                    model_path = self.storage_path / "models" / f"{safe_key}.json"
                    model.save(str(model_path))
            
            models_path = self.storage_path / "models" / "traditional_models.json"
            with open(models_path, 'w', encoding='utf-8') as f:
                json.dump(self.models, f, indent=2, ensure_ascii=False, default=str)
            
            print("💾 All models saved successfully")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
    
    def load_models(self):
        try:
            models_dir = self.storage_path / "models"
            if not models_dir.exists():
                return
            
            for model_file in models_dir.glob("*.json"):
                if model_file.name != "traditional_models.json":
                    try:
                        model = NeuralNetwork(layers=[1], activation='relu')
                        model.load(str(model_file))
                        
                        model_key = model_file.stem
                        self.deep_models[model_key] = model
                        
                        print(f"✅ Loaded deep model: {model_key}")
                    except Exception as e:
                        print(f"❌ Error loading model {model_file}: {e}")
            
            trad_path = models_dir / "traditional_models.json"
            if trad_path.exists():
                with open(trad_path, 'r', encoding='utf-8') as f:
                    self.models = json.load(f)
                print(f"✅ Loaded {len(self.models)} traditional models")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        stats = {
            'deep_models': {},
            'traditional_models': {},
            'total_deep_models': len(self.deep_models),
            'total_traditional_models': len(self.models),
            'total_samples': 0
        }
        
        for key, model in self.deep_models.items():
            if hasattr(model, 'layers'):
                stats['deep_models'][key] = {
                    'layers': model.layers,
                    'activation': model.activation,
                    'output_activation': model.output_activation,
                    'parameters': sum(len(w) * len(w[0]) for w in model.weights) + sum(len(b[0]) for b in model.biases)
                }
        
        for key, model in self.models.items():
            stats['traditional_models'][key] = {
                'samples': len(model.get('samples', [])),
                'labels': list(model.get('label_stats', {}).keys()),
                'last_updated': model.get('last_updated', 'unknown')
            }
            stats['total_samples'] += len(model.get('samples', []))
        
        return stats
    
    def stop(self):
        self.is_running = False
        self.save_models()

# ============================================================================
# هسته مرکزی ImanCore
# ============================================================================

class ImanCore:
    def __init__(self):
        print("🚀 Starting ImanCore v3.0...")
        print("=" * 60)
        
        self.api_manager = APIKeyManager()
        self.file_processor = FileProcessor()
        self.learning_engine = LearningEngine()
        
        self.db_path = "imancore.db"
        self._init_database()
        
        self.settings = self._load_settings()
        self.admin_api_key = None
        self._ensure_admin_key()
        
        print("✅ ImanCore initialized successfully!")
        print("=" * 60)
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                hash_md5 TEXT NOT NULL,
                uploaded_by TEXT NOT NULL,
                upload_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                metadata TEXT,
                storage_path TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_records (
                record_id TEXT PRIMARY KEY,
                source_app TEXT NOT NULL,
                data_type TEXT NOT NULL,
                content TEXT,
                features TEXT,
                labels TEXT,
                confidence REAL DEFAULT 0.0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                log_id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                app_name TEXT,
                action TEXT,
                details TEXT,
                status TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_settings(self) -> Dict[str, Any]:
        settings_path = "imancore_settings.json"
        default_settings = {
            'max_file_size_mb': 100,
            'allowed_file_types': ['txt', 'csv', 'json', 'pdf', 'jpg', 'png', 'zip'],
            'learning_enabled': True,
            'deep_learning_enabled': True,
            'auto_backup_hours': 24,
            'max_api_requests_per_hour': 1000,
            'storage_path': './imancore_storage',
            'backup_path': './imancore_backups',
            'log_level': 'INFO'
        }
        
        try:
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    default_settings.update(loaded_settings)
        except Exception as e:
            print(f"⚠️ Error loading settings: {e}")
        
        return default_settings
    
    def _save_settings(self):
        settings_path = "imancore_settings.json"
        try:
            with open(settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
    
    def _ensure_admin_key(self):
        conn = sqlite3.connect(self.api_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT api_key FROM api_keys WHERE owner = ? AND app_name = ?', 
                      ('admin', 'iman_system'))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            key_id, api_key = self.api_manager.generate_api_key(
                app_name='iman_system',
                owner='admin',
                permissions=['upload', 'learn', 'predict', 'query', 'admin', 'backup']
            )
            self.admin_api_key = api_key
            print(f"🔑 Admin API Key created: {key_id}")
        else:
            self.admin_api_key = row[0]
    
    def log_activity(self, app_name: str, action: str, details: str = "", 
                    status: str = "success") -> str:
        log_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO activity_log (log_id, app_name, action, details, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (log_id, app_name, action, details, status))
        
        conn.commit()
        conn.close()
        
        return log_id
    
    def register_app(self, app_name: str, owner: str, 
                    permissions: List[str] = None) -> Dict[str, Any]:
        if permissions is None:
            permissions = ['upload', 'learn', 'predict', 'query']
        
        key_id, api_key = self.api_manager.generate_api_key(app_name, owner, permissions)
        
        self.log_activity(app_name, "register_app", 
                         f"Application registered: {app_name}, Owner: {owner}")
        
        return {
            'success': True,
            'key_id': key_id,
            'api_key': api_key,
            'app_name': app_name,
            'permissions': permissions,
            'message': 'Application registered successfully'
        }
    
    def upload_file(self, api_key: str, filename: str, content: bytes) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'upload')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        max_size = self.settings['max_file_size_mb'] * 1024 * 1024
        if len(content) > max_size:
            return {'error': f'File too large. Maximum size is {self.settings["max_file_size_mb"]}MB'}
        
        file_type = self.file_processor.detect_file_type(filename)
        if file_type == FileType.UNKNOWN:
            return {'error': 'File type not supported'}
        
        file_id = str(uuid.uuid4())
        storage_path, saved_file_type = self.file_processor.save_file(file_id, filename, content)
        
        metadata = self.file_processor.extract_metadata(storage_path, saved_file_type)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO files 
            (file_id, original_name, file_type, size_bytes, hash_md5, 
             uploaded_by, metadata, storage_path, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_id,
            filename,
            saved_file_type.value,
            len(content),
            self.file_processor.calculate_hash(content),
            key_info.key_id,
            json.dumps(metadata),
            storage_path,
            False
        ))
        
        conn.commit()
        conn.close()
        
        self.log_activity(key_info.app_name, "upload_file", 
                         f"File uploaded: {filename} ({len(content)} bytes)")
        
        return {
            'success': True,
            'file_id': file_id,
            'filename': filename,
            'file_type': saved_file_type.value,
            'size_bytes': len(content),
            'hash_md5': self.file_processor.calculate_hash(content),
            'upload_time': datetime.datetime.now().isoformat(),
            'metadata': metadata,
            'download_url': f"/files/{file_id}"
        }
    
    def add_learning(self, api_key: str, data_type: str, content: str,
                    labels: List[str] = None) -> Dict[str, Any]:
        if labels is None:
            labels = []
        
        key_info = self.api_manager.validate_api_key(api_key, 'learn')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        record_id = str(uuid.uuid4())
        
        features = self.learning_engine._extract_features(content, data_type)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_records 
            (record_id, source_app, data_type, content, features, labels)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            record_id,
            key_info.key_id,
            data_type,
            content,
            json.dumps(features),
            json.dumps(labels)
        ))
        
        conn.commit()
        conn.close()
        
        self.learning_engine.add_learning_record(
            key_info.key_id, data_type, content, labels
        )
        
        self.log_activity(key_info.app_name, "add_learning", 
                         f"Learning data added: {data_type}, Labels: {labels}")
        
        return {
            'success': True,
            'record_id': record_id,
            'data_type': data_type,
            'labels': labels,
            'features': features,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def predict(self, api_key: str, model_key: str, data_type: str, 
               content: str) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'predict')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        prediction = self.learning_engine.predict(key_info.key_id, data_type, content)
        
        self.log_activity(key_info.app_name, "predict", 
                         f"Prediction requested: {model_key}, Type: {data_type}")
        
        return {
            'success': True,
            'app_name': key_info.app_name,
            'model_key': model_key,
            'data_type': data_type,
            'prediction': prediction,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    def train_deep_model(self, api_key: str, model_key: str, 
                        X: List[List[float]], y: List[List[float]],
                        epochs: int = 100, layers: List[int] = None) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'learn')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        result = self.learning_engine.train_deep_model_custom(
            f"{key_info.key_id}_{model_key}", X, y, epochs, layers
        )
        
        self.log_activity(key_info.app_name, "train_deep_model", 
                         f"Deep model trained: {model_key}, Epochs: {epochs}")
        
        return result
    
    def query_files(self, api_key: str, filters: Dict = None) -> List[Dict]:
        if filters is None:
            filters = {}
        
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return [{'error': 'Invalid API Key or insufficient permissions'}]
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM files WHERE uploaded_by = ?"
        params = [key_info.key_id]
        
        if 'file_type' in filters:
            query += " AND file_type = ?"
            params.append(filters['file_type'])
        
        if 'start_date' in filters:
            query += " AND upload_time >= ?"
            params.append(filters['start_date'])
        
        if 'end_date' in filters:
            query += " AND upload_time <= ?"
            params.append(filters['end_date'])
        
        if 'search' in filters:
            query += " AND original_name LIKE ?"
            params.append(f'%{filters["search"]}%')
        
        if 'processed' in filters:
            query += " AND processed = ?"
            params.append(1 if filters['processed'] else 0)
        
        query += " ORDER BY upload_time DESC LIMIT 100"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        files = []
        for row in rows:
            files.append({
                'file_id': row['file_id'],
                'original_name': row['original_name'],
                'file_type': row['file_type'],
                'size_bytes': row['size_bytes'],
                'hash_md5': row['hash_md5'],
                'upload_time': row['upload_time'],
                'processed': bool(row['processed']),
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'storage_path': row['storage_path']
            })
        
        self.log_activity(key_info.app_name, "query_files", 
                         f"Files queried: {len(files)} files found")
        
        return files
    
    def get_file_content(self, api_key: str, file_id: str) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM files 
            WHERE file_id = ? AND uploaded_by = ?
        ''', (file_id, key_info.key_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return {'error': 'File not found or access denied'}
        
        try:
            file_type = FileType(row['file_type'])
            content = self.file_processor.extract_text_from_file(
                row['storage_path'], file_type
            )
            
            self.log_activity(key_info.app_name, "get_file_content", 
                            f"File content retrieved: {row['original_name']}")
            
            return {
                'success': True,
                'file_id': row['file_id'],
                'filename': row['original_name'],
                'file_type': row['file_type'],
                'content': content[:5000],
                'full_content_available': len(content) <= 5000,
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            }
            
        except Exception as e:
            return {'error': f'Error reading file: {str(e)}'}
    
    def get_activity_log(self, api_key: str, limit: int = 100, 
                        filters: Dict = None) -> List[Dict]:
        if filters is None:
            filters = {}
        
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return [{'error': 'Invalid API Key or insufficient permissions'}]
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM activity_log WHERE app_name = ?"
        params = [key_info.app_name]
        
        if 'action' in filters:
            query += " AND action = ?"
            params.append(filters['action'])
        
        if 'status' in filters:
            query += " AND status = ?"
            params.append(filters['status'])
        
        if 'start_date' in filters:
            query += " AND timestamp >= ?"
            params.append(filters['start_date'])
        
        if 'end_date' in filters:
            query += " AND timestamp <= ?"
            params.append(filters['end_date'])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        activities = []
        for row in rows:
            activities.append({
                'log_id': row['log_id'],
                'timestamp': row['timestamp'],
                'action': row['action'],
                'details': row['details'],
                'status': row['status']
            })
        
        return activities
    
    def get_stats(self, api_key: str) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total_files,
                   SUM(size_bytes) as total_size,
                   file_type,
                   COUNT(*) as type_count
            FROM files 
            WHERE uploaded_by = ?
            GROUP BY file_type
        ''', (key_info.key_id,))
        
        file_stats = cursor.fetchall()
        
        cursor.execute('''
            SELECT COUNT(*) as total_records,
                   data_type,
                   COUNT(*) as type_count
            FROM learning_records 
            WHERE source_app = ?
            GROUP BY data_type
        ''', (key_info.key_id,))
        
        learning_stats = cursor.fetchall()
        
        cursor.execute('''
            SELECT COUNT(*) as total_activities,
                   status,
                   COUNT(*) as status_count
            FROM activity_log 
            WHERE app_name = ?
            GROUP BY status
        ''', (key_info.app_name,))
        
        activity_stats = cursor.fetchall()
        
        conn.close()
        
        model_stats = self.learning_engine.get_model_stats()
        api_stats = self.api_manager.get_key_stats()
        storage_usage = self._get_storage_usage()
        
        return {
            'app_name': key_info.app_name,
            'owner': key_info.owner,
            'permissions': key_info.permissions,
            'total_requests': key_info.total_requests,
            
            'files': {
                'total': sum(row[0] for row in file_stats) if file_stats else 0,
                'total_size_mb': sum(row[1] for row in file_stats) / (1024*1024) if file_stats else 0,
                'by_type': {row[2]: row[3] for row in file_stats}
            },
            
            'learning': {
                'total': sum(row[0] for row in learning_stats) if learning_stats else 0,
                'by_type': {row[1]: row[2] for row in learning_stats}
            },
            
            'activities': {
                'total': sum(row[0] for row in activity_stats) if activity_stats else 0,
                'by_status': {row[1]: row[2] for row in activity_stats}
            },
            
            'models': model_stats,
            'api_stats': api_stats,
            'storage_usage_mb': storage_usage,
            
            'system': {
                'timestamp': datetime.datetime.now().isoformat(),
                'version': '3.0.0',
                'deep_learning_enabled': self.settings['deep_learning_enabled'],
                'learning_enabled': self.settings['learning_enabled']
            }
        }
    
    def _get_storage_usage(self) -> float:
        try:
            total_size = 0
            storage_path = Path(self.settings['storage_path'])
            
            if storage_path.exists():
                for file_path in storage_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
            
            return total_size / (1024 * 1024)
        except:
            return 0.0
    
    def export_data(self, api_key: str, export_type: str = 'all') -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        export_id = str(uuid.uuid4())
        export_dir = Path(self.settings['storage_path']) / "exports" / export_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            export_data = {
                'export_id': export_id,
                'app_name': key_info.app_name,
                'timestamp': datetime.datetime.now().isoformat(),
                'files': []
            }
            
            if export_type in ['all', 'files']:
                files = self.query_files(api_key)
                export_data['files'] = files
                
                files_path = export_dir / "files.json"
                with open(files_path, 'w', encoding='utf-8') as f:
                    json.dump(files, f, indent=2, ensure_ascii=False)
            
            if export_type in ['all', 'learning']:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM learning_records 
                    WHERE source_app = ?
                ''', (key_info.key_id,))
                
                learning_records = [dict(row) for row in cursor.fetchall()]
                conn.close()
                
                export_data['learning_records'] = learning_records
                
                learning_path = export_dir / "learning_records.json"
                with open(learning_path, 'w', encoding='utf-8') as f:
                    json.dump(learning_records, f, indent=2, default=str, ensure_ascii=False)
            
            if export_type in ['all', 'models']:
                model_stats = self.learning_engine.get_model_stats()
                export_data['models'] = model_stats
                
                models_path = export_dir / "models.json"
                with open(models_path, 'w', encoding='utf-8') as f:
                    json.dump(model_stats, f, indent=2, ensure_ascii=False)
            
            zip_path = export_dir.parent / f"{export_id}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.glob("*.json"):
                    zipf.write(file_path, file_path.name)
            
            shutil.rmtree(export_dir)
            
            self.log_activity(key_info.app_name, "export_data", 
                            f"Data exported: {export_type}")
            
            return {
                'success': True,
                'export_id': export_id,
                'download_url': f"/exports/{export_id}.zip",
                'size_bytes': os.path.getsize(zip_path),
                'export_type': export_type,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Export failed: {str(e)}'}
    
    def backup_system(self, api_key: str) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'query')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        if key_info.owner != 'admin':
            return {'error': 'Only system owner can create backups'}
        
        backup_id = str(uuid.uuid4())
        backup_dir = Path(self.settings.get('backup_path', './imancore_backups'))
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_folder = backup_dir / f"backup_{timestamp}_{backup_id[:8]}"
            backup_folder.mkdir(parents=True, exist_ok=True)
            
            databases = ['imancore.db', 'imancore_keys.db']
            for db_name in databases:
                if os.path.exists(db_name):
                    shutil.copy2(db_name, backup_folder / db_name)
            
            storage_path = Path(self.settings['storage_path'])
            if storage_path.exists():
                backup_storage = backup_folder / "storage"
                shutil.copytree(storage_path, backup_storage)
            
            settings_path = "imancore_settings.json"
            if os.path.exists(settings_path):
                shutil.copy2(settings_path, backup_folder / "settings.json")
            
            backup_report = {
                'backup_id': backup_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'created_by': key_info.app_name,
                'databases': databases,
                'storage_path': str(storage_path),
                'total_size_mb': self._get_folder_size(backup_folder) / (1024*1024)
            }
            
            with open(backup_folder / "backup_report.json", 'w', encoding='utf-8') as f:
                json.dump(backup_report, f, indent=2, ensure_ascii=False)
            
            zip_path = backup_dir / f"backup_{timestamp}_{backup_id[:8]}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in backup_folder.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(backup_folder)
                        zipf.write(file_path, arcname)
            
            shutil.rmtree(backup_folder)
            
            self.log_activity(key_info.app_name, "backup_system", 
                            f"System backup created: {backup_id}")
            
            return {
                'success': True,
                'backup_id': backup_id,
                'backup_path': str(zip_path),
                'size_mb': os.path.getsize(zip_path) / (1024*1024),
                'timestamp': datetime.datetime.now().isoformat(),
                'report': backup_report
            }
            
        except Exception as e:
            return {'error': f'Backup failed: {str(e)}'}
    
    def _get_folder_size(self, folder_path: Path) -> int:
        total_size = 0
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def update_settings(self, api_key: str, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        key_info = self.api_manager.validate_api_key(api_key, 'admin')
        if not key_info:
            return {'error': 'Invalid API Key or insufficient permissions'}
        
        if key_info.owner != 'admin':
            return {'error': 'Only system owner can update settings'}
        
        valid_settings = [
            'max_file_size_mb', 'allowed_file_types', 'learning_enabled',
            'deep_learning_enabled', 'auto_backup_hours', 'max_api_requests_per_hour',
            'storage_path', 'backup_path', 'log_level'
        ]
        
        for key, value in new_settings.items():
            if key in valid_settings:
                self.settings[key] = value
        
        self._save_settings()
        
        self.log_activity(key_info.app_name, "update_settings", 
                         f"Settings updated: {list(new_settings.keys())}")
        
        return {
            'success': True,
            'message': 'Settings updated successfully',
            'updated_settings': new_settings
        }
    
    def shutdown(self):
        print("🔴 Shutting down ImanCore...")
        self.learning_engine.save_models()
        self.learning_engine.stop()
        self._save_settings()
        print("✅ ImanCore shutdown complete")

# ============================================================================
# API Server با FastAPI
# ============================================================================

if FASTAPI_AVAILABLE:
    # مدل‌های Pydantic برای API
    class RegisterAppRequest(BaseModel):
        app_name: str = Field(..., min_length=1, max_length=100)
        owner: str = Field(..., min_length=1, max_length=100)
        permissions: List[str] = Field(default=["upload", "learn", "predict", "query"])
    
    class UploadFileRequest(BaseModel):
        filename: str = Field(..., min_length=1)
    
    class AddLearningRequest(BaseModel):
        data_type: str = Field(..., min_length=1)
        content: str = Field(..., min_length=1)
        labels: List[str] = Field(default=[])
    
    class PredictRequest(BaseModel):
        model_key: str = Field(..., min_length=1)
        data_type: str = Field(..., min_length=1)
        content: str = Field(..., min_length=1)
    
    class TrainModelRequest(BaseModel):
        model_key: str = Field(..., min_length=1)
        X: List[List[float]]
        y: List[List[float]]
        epochs: int = Field(default=100, ge=1, le=10000)
        layers: Optional[List[int]] = None
    
    class QueryFilesRequest(BaseModel):
        filters: Dict[str, Any] = Field(default={})
    
    class ExportDataRequest(BaseModel):
        export_type: str = Field(default="all", pattern="^(all|files|learning|models)$")  # تغییر regex به pattern
    
    class UpdateSettingsRequest(BaseModel):
        settings: Dict[str, Any]
    
    # وابستگی‌های API
    def get_api_key(api_key: str = Header(..., alias="X-API-Key")):
        return api_key
    
    def verify_permission(api_key: str = Depends(get_api_key), permission: str = None):
        core = get_core()
        key_info = core.api_manager.validate_api_key(api_key, permission)
        if not key_info:
            raise HTTPException(status_code=401, detail="Invalid API Key or insufficient permissions")
        return key_info
    
    # Singleton برای ImanCore
    _core_instance = None
    
    def get_core():
        global _core_instance
        if _core_instance is None:
            _core_instance = ImanCore()
        return _core_instance
    
    # ایجاد FastAPI App
    app = FastAPI(
        title="ImanCore API",
        description="هسته مرکزی هوش مصنوعی با دیپ لرنینگ داخلی",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Endpoint‌های API
    @app.post("/register", tags=["Authentication"])
    async def register_app(request: RegisterAppRequest):
        """ثبت اپلیکیشن جدید و دریافت API Key"""
        core = get_core()
        result = core.register_app(request.app_name, request.owner, request.permissions)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.post("/upload", tags=["Files"])
    async def upload_file(
        api_key: str = Depends(get_api_key),
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None
    ):
        """آپلود فایل"""
        core = get_core()
        
        try:
            content = await file.read()
            result = core.upload_file(api_key, file.filename, content)
            
            if 'error' in result:
                raise HTTPException(status_code=400, detail=result['error'])
            
            return JSONResponse(content=result)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/learn", tags=["Learning"])
    async def add_learning(
        request: AddLearningRequest,
        api_key: str = Depends(get_api_key)
    ):
        """افزودن داده برای یادگیری"""
        core = get_core()
        result = core.add_learning(api_key, request.data_type, request.content, request.labels)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.post("/predict", tags=["Prediction"])
    async def predict(
        request: PredictRequest,
        api_key: str = Depends(get_api_key)
    ):
        """درخواست پیش‌بینی"""
        core = get_core()
        result = core.predict(api_key, request.model_key, request.data_type, request.content)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.post("/train", tags=["Deep Learning"])
    async def train_model(
        request: TrainModelRequest,
        api_key: str = Depends(get_api_key)
    ):
        """آموزش مدل دیپ لرنینگ"""
        core = get_core()
        result = core.train_deep_model(
            api_key, request.model_key, request.X, request.y, request.epochs, request.layers
        )
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.get("/files", tags=["Files"])
    async def get_files(
        api_key: str = Depends(get_api_key),
        file_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=1000)
    ):
        """لیست فایل‌ها"""
        core = get_core()
        
        filters = {}
        if file_type:
            filters['file_type'] = file_type
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        if search:
            filters['search'] = search
        
        result = core.query_files(api_key, filters)
        
        if result and 'error' in result[0]:
            raise HTTPException(status_code=400, detail=result[0]['error'])
        
        return JSONResponse(content=result[:limit])
    
    @app.get("/files/{file_id}", tags=["Files"])
    async def get_file(
        file_id: str,
        api_key: str = Depends(get_api_key)
    ):
        """دریافت محتوای فایل"""
        core = get_core()
        result = core.get_file_content(api_key, file_id)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.get("/activity", tags=["Monitoring"])
    async def get_activity(
        api_key: str = Depends(get_api_key),
        action: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = Query(default=100, ge=1, le=1000)
    ):
        """دریافت لاگ فعالیت‌ها"""
        core = get_core()
        
        filters = {}
        if action:
            filters['action'] = action
        if status:
            filters['status'] = status
        if start_date:
            filters['start_date'] = start_date
        if end_date:
            filters['end_date'] = end_date
        
        result = core.get_activity_log(api_key, limit, filters)
        
        if result and 'error' in result[0]:
            raise HTTPException(status_code=400, detail=result[0]['error'])
        
        return JSONResponse(content=result)
    
    @app.get("/stats", tags=["Monitoring"])
    async def get_stats(api_key: str = Depends(get_api_key)):
        """دریافت آمار سیستم"""
        core = get_core()
        result = core.get_stats(api_key)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.post("/export", tags=["Data Management"])
    async def export_data(
        request: ExportDataRequest,
        api_key: str = Depends(get_api_key)
    ):
        """خروجی گرفتن از داده‌ها"""
        core = get_core()
        result = core.export_data(api_key, request.export_type)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.post("/backup", tags=["System"])
    async def backup_system(api_key: str = Depends(get_api_key)):
        """پشتیبان‌گیری از سیستم"""
        core = get_core()
        result = core.backup_system(api_key)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.put("/settings", tags=["System"])
    async def update_settings(
        request: UpdateSettingsRequest,
        api_key: str = Depends(get_api_key)
    ):
        """به‌روزرسانی تنظیمات سیستم"""
        core = get_core()
        result = core.update_settings(api_key, request.settings)
        
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return JSONResponse(content=result)
    
    @app.get("/health", tags=["System"])
    async def health_check():
        """بررسی سلامت سیستم"""
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "3.0.0"
        })
    
    @app.get("/download/{export_id}", tags=["Files"])
    async def download_export(export_id: str):
        """دانلود فایل خروجی"""
        export_path = Path("./imancore_storage/exports") / f"{export_id}.zip"
        
        if not export_path.exists():
            raise HTTPException(status_code=404, detail="Export not found")
        
        return FileResponse(
            path=export_path,
            filename=f"export_{export_id}.zip",
            media_type="application/zip"
        )
    
    def run_api_server(host: str = "0.0.0.0", port: int = 8000):
        """اجرای API سرور"""
        print(f"🌐 Starting API Server on http://{host}:{port}")
        print(f"📚 API Documentation: http://{host}:{port}/docs")
        print(f"📖 ReDoc: http://{host}:{port}/redoc")
        
        uvicorn.run(app, host=host, port=port)

# ============================================================================
# GUI با PyQt5
# ============================================================================

if PYQT_AVAILABLE:
    class LoadingDialog(QDialog):
        def __init__(self, title: str = "در حال پردازش...", parent=None):
            super().__init__(parent)
            self.setWindowTitle(title)
            self.setModal(True)
            self.setFixedSize(400, 150)
            
            layout = QVBoxLayout()
            
            self.label = QLabel("لطفاً منتظر بمانید...")
            self.label.setAlignment(Qt.AlignCenter)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            
            self.cancel_btn = QPushButton("لغو")
            self.cancel_btn.clicked.connect(self.reject)
            
            layout.addWidget(self.label)
            layout.addWidget(self.progress_bar)
            layout.addWidget(self.cancel_btn, 0, Qt.AlignCenter)
            
            self.setLayout(layout)
        
        def update_progress(self, value: int, message: str = ""):
            self.progress_bar.setValue(value)
            if message:
                self.label.setText(message)
    
    class CreateApiKeyDialog(QDialog):
        def __init__(self, core: ImanCore, parent=None):
            super().__init__(parent)
            self.core = core
            self.setWindowTitle("ایجاد API Key جدید")
            self.setFixedSize(500, 400)
            
            layout = QVBoxLayout()
            
            form_layout = QFormLayout()
            
            self.app_name_input = QLineEdit()
            self.app_name_input.setPlaceholderText("نام اپلیکیشن")
            
            self.owner_input = QLineEdit()
            self.owner_input.setPlaceholderText("نام مالک")
            self.owner_input.setText("user")
            
            self.permissions_input = QTextEdit()
            self.permissions_input.setMaximumHeight(100)
            self.permissions_input.setPlainText("upload,learn,predict,query")
            self.permissions_input.setToolTip("مجوزها را با کاما جدا کنید")
            
            self.rate_limit_input = QSpinBox()
            self.rate_limit_input.setRange(1, 10000)
            self.rate_limit_input.setValue(1000)
            self.rate_limit_input.setSuffix(" درخواست در ساعت")
            
            form_layout.addRow("نام اپلیکیشن:", self.app_name_input)
            form_layout.addRow("مالک:", self.owner_input)
            form_layout.addRow("مجوزها:", self.permissions_input)
            form_layout.addRow("محدودیت نرخ:", self.rate_limit_input)
            
            layout.addLayout(form_layout)
            
            button_box = QDialogButtonBox(
                QDialogButtonBox.Ok | 
                QDialogButtonBox.Cancel
            )
            button_box.accepted.connect(self.create_api_key)
            button_box.rejected.connect(self.reject)
            
            layout.addWidget(button_box)
            
            self.setLayout(layout)
        
        def create_api_key(self):
            app_name = self.app_name_input.text().strip()
            owner = self.owner_input.text().strip()
            permissions_text = self.permissions_input.toPlainText().strip()
            rate_limit = self.rate_limit_input.value()
            
            if not app_name or not owner:
                QMessageBox.warning(self, "خطا", "لطفاً نام اپلیکیشن و مالک را وارد کنید")
                return
            
            permissions = [p.strip() for p in permissions_text.split(',') if p.strip()]
            
            try:
                result = self.core.register_app(app_name, owner, permissions)
                
                if result.get('success'):
                    api_key = result['api_key']
                    key_id = result['key_id']
                    
                    msg = f"""✅ API Key ایجاد شد!

🔑 Key ID: {key_id}
📱 Application: {app_name}
👤 Owner: {owner}
📋 Permissions: {', '.join(permissions)}
🚦 Rate Limit: {rate_limit}/hour

⚠️ **API Key را ذخیره کنید، دیگر نمایش داده نخواهد شد:**
{api_key}"""
                    
                    QMessageBox.information(self, "API Key ایجاد شد", msg)
                    self.accept()
                else:
                    QMessageBox.critical(self, "خطا", result.get('error', 'خطای ناشناخته'))
            
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در ایجاد API Key: {str(e)}")
    
    class BackgroundWorker(QThread):
        progress = pyqtSignal(int)
        message = pyqtSignal(str)
        finished = pyqtSignal(dict)
        
        def __init__(self, task_type: str, core: ImanCore, **kwargs):
            super().__init__()
            self.task_type = task_type
            self.core = core
            self.kwargs = kwargs
            self.is_running = True
        
        def run(self):
            try:
                if self.task_type == 'train_model':
                    self._train_model_task()
                elif self.task_type == 'export_data':
                    self._export_data_task()
                elif self.task_type == 'backup':
                    self._backup_task()
                elif self.task_type == 'process_files':
                    self._process_files_task()
                
            except Exception as e:
                self.finished.emit({'success': False, 'error': str(e)})
        
        def stop(self):
            self.is_running = False
        
        def _train_model_task(self):
            api_key = self.kwargs.get('api_key')
            model_key = self.kwargs.get('model_key')
            X = self.kwargs.get('X')
            y = self.kwargs.get('y')
            epochs = self.kwargs.get('epochs', 100)
            
            self.progress.emit(5)
            self.message.emit("در حال آماده‌سازی داده‌ها...")
            
            result = self.core.train_deep_model(api_key, model_key, X, y, epochs)
            
            if result.get('success'):
                self.progress.emit(100)
                self.message.emit("✅ آموزش مدل تکمیل شد")
                self.finished.emit(result)
            else:
                self.finished.emit({'success': False, 'error': result.get('error', 'Unknown error')})
        
        def _export_data_task(self):
            api_key = self.kwargs.get('api_key')
            export_type = self.kwargs.get('export_type', 'all')
            
            self.progress.emit(10)
            self.message.emit("در حال جمع‌آوری داده‌ها...")
            
            self.progress.emit(30)
            self.message.emit("در حال ایجاد فایل‌های خروجی...")
            
            result = self.core.export_data(api_key, export_type)
            
            if result.get('success'):
                self.progress.emit(100)
                self.message.emit("✅ خروجی داده‌ها تکمیل شد")
                self.finished.emit(result)
            else:
                self.finished.emit({'success': False, 'error': result.get('error', 'Export failed')})
        
        def _backup_task(self):
            api_key = self.kwargs.get('api_key')
            
            self.progress.emit(20)
            self.message.emit("در حال پشتیبان‌گیری از دیتابیس...")
            
            self.progress.emit(50)
            self.message.emit("در حال پشتیبان‌گیری از فایل‌ها...")
            
            result = self.core.backup_system(api_key)
            
            if result.get('success'):
                self.progress.emit(100)
                self.message.emit("✅ پشتیبان‌گیری تکمیل شد")
                self.finished.emit(result)
            else:
                self.finished.emit({'success': False, 'error': result.get('error', 'Backup failed')})
        
        def _process_files_task(self):
            file_paths = self.kwargs.get('file_paths', [])
            api_key = self.kwargs.get('api_key')
            
            total_files = len(file_paths)
            processed_files = []
            
            for i, file_path in enumerate(file_paths):
                if not self.is_running:
                    break
                
                progress = int((i / total_files) * 100)
                self.progress.emit(progress)
                
                filename = Path(file_path).name
                self.message.emit(f"در حال پردازش فایل {i+1}/{total_files}: {filename}")
                
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    
                    result = self.core.upload_file(api_key, filename, content)
                    
                    if 'success' in result and result['success']:
                        processed_files.append({
                            'filename': filename,
                            'file_id': result.get('file_id'),
                            'success': True
                        })
                    else:
                        processed_files.append({
                            'filename': filename,
                            'error': result.get('error', 'Upload failed'),
                            'success': False
                        })
                    
                except Exception as e:
                    processed_files.append({
                        'filename': filename,
                        'error': str(e),
                        'success': False
                    })
                
                time.sleep(0.1)
            
            self.progress.emit(100)
            self.message.emit(f"✅ پردازش {len(processed_files)} فایل تکمیل شد")
            
            self.finished.emit({
                'success': True,
                'processed_files': processed_files,
                'total_files': total_files,
                'successful_uploads': sum(1 for f in processed_files if f.get('success', False))
            })
    
    class ImanCoreGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            
            print("🚀 Launching ImanCore GUI...")
            
            self.core = ImanCore()
            
            self.setWindowTitle("🧠 ImanCore v3.0 - هسته مرکزی هوش مصنوعی")
            self.setGeometry(100, 50, 1600, 900)
            
            self.setWindowIcon(self._create_icon())
            self._setup_ui()
            self._load_initial_data()
            
            print("✅ ImanCore GUI is ready!")
        
        def _create_icon(self) -> QIcon:
            pixmap = QPixmap(64, 64)
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            painter.setBrush(QBrush(QColor(138, 43, 226)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(4, 4, 56, 56)
            
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 24, QFont.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "IC")
            
            painter.end()
            
            return QIcon(pixmap)
        
        def _setup_ui(self):
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)
            
            self._create_header(main_layout)
            
            self.tab_widget = QTabWidget()
            
            self.tab_dashboard = self._create_dashboard_tab()
            self.tab_api_keys = self._create_api_keys_tab()
            self.tab_files = self._create_files_tab()
            self.tab_learning = self._create_learning_tab()
            self.tab_deep_learning = self._create_deep_learning_tab()
            self.tab_stats = self._create_stats_tab()
            self.tab_activity = self._create_activity_tab()
            self.tab_settings = self._create_settings_tab()
            
            self.tab_widget.addTab(self.tab_dashboard, "🏠 داشبورد")
            self.tab_widget.addTab(self.tab_api_keys, "🔑 API Keys")
            self.tab_widget.addTab(self.tab_files, "📁 فایل‌ها")
            self.tab_widget.addTab(self.tab_learning, "🧠 یادگیری")
            self.tab_widget.addTab(self.tab_deep_learning, "🤖 دیپ لرنینگ")
            self.tab_widget.addTab(self.tab_stats, "📊 آمار")
            self.tab_widget.addTab(self.tab_activity, "📋 فعالیت‌ها")
            self.tab_widget.addTab(self.tab_settings, "⚙️ تنظیمات")
            
            main_layout.addWidget(self.tab_widget, 1)
            
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            
            self.status_label = QLabel("🟢 سیستم آماده است")
            self.status_bar.addWidget(self.status_label)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximumWidth(200)
            self.progress_bar.setVisible(False)
            self.status_bar.addPermanentWidget(self.progress_bar)
            
            self._create_menu()
        
        def _create_header(self, parent_layout):
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            
            logo_label = QLabel("🧠")
            logo_label.setFont(QFont("Arial", 32))
            logo_label.setFixedWidth(60)
            
            title_layout = QVBoxLayout()
            
            title_label = QLabel("ImanCore v3.0")
            title_label.setFont(QFont("Vazir", 20, QFont.Bold))
            title_label.setStyleSheet("color: #8B2BE2;")
            
            subtitle_label = QLabel("هسته مرکزی هوش مصنوعی با دیپ لرنینگ داخلی")
            subtitle_label.setFont(QFont("Vazir", 11))
            subtitle_label.setStyleSheet("color: #666;")
            
            title_layout.addWidget(title_label)
            title_layout.addWidget(subtitle_label)
            
            header_layout.addWidget(logo_label)
            header_layout.addLayout(title_layout)
            header_layout.addStretch()
            
            quick_actions = QHBoxLayout()
            quick_actions.setSpacing(5)
            
            self.refresh_btn = QPushButton("🔄 به‌روزرسانی")
            self.refresh_btn.clicked.connect(self.refresh_all)
            
            self.api_key_btn = QPushButton("🔑 مدیریت API Keys")
            self.api_key_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
            
            self.upload_btn = QPushButton("📤 آپلود فایل")
            self.upload_btn.clicked.connect(self.show_upload_dialog)
            
            self.train_btn = QPushButton("🤖 آموزش مدل")
            self.train_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(4))
            
            self.stats_btn = QPushButton("📊 آمار سیستم")
            self.stats_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(5))
            
            quick_actions.addWidget(self.refresh_btn)
            quick_actions.addWidget(self.api_key_btn)
            quick_actions.addWidget(self.upload_btn)
            quick_actions.addWidget(self.train_btn)
            quick_actions.addWidget(self.stats_btn)
            
            header_layout.addLayout(quick_actions)
            
            parent_layout.addWidget(header_widget)
        
        def _create_menu(self):
            menubar = self.menuBar()
            
            file_menu = menubar.addMenu("📁 فایل")
            
            new_api_action = QAction("➕ ایجاد API Key جدید", self)
            new_api_action.triggered.connect(self.show_create_api_dialog)
            file_menu.addAction(new_api_action)
            
            upload_action = QAction("📤 آپلود فایل...", self)
            upload_action.triggered.connect(self.show_upload_dialog)
            file_menu.addAction(upload_action)
            
            file_menu.addSeparator()
            
            export_action = QAction("📊 خروجی داده‌ها...", self)
            export_action.triggered.connect(self.show_export_dialog)
            file_menu.addAction(export_action)
            
            backup_action = QAction("💾 پشتیبان‌گیری...", self)
            backup_action.triggered.connect(self.show_backup_dialog)
            file_menu.addAction(backup_action)
            
            file_menu.addSeparator()
            
            exit_action = QAction("🚪 خروج", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            learn_menu = menubar.addMenu("🧠 یادگیری")
            
            add_learn_action = QAction("➕ افزودن داده آموزشی...", self)
            add_learn_action.triggered.connect(self.show_add_learning_dialog)
            learn_menu.addAction(add_learn_action)
            
            train_dl_action = QAction("🤖 آموزش مدل عمیق...", self)
            train_dl_action.triggered.connect(self.show_train_dl_dialog)
            learn_menu.addAction(train_dl_action)
            
            predict_action = QAction("🔮 پیش‌بینی...", self)
            predict_action.triggered.connect(self.show_predict_dialog)
            learn_menu.addAction(predict_action)
            
            tools_menu = menubar.addMenu("⚙️ ابزارها")
            
            restart_action = QAction("🔄 راه‌اندازی مجدد هسته", self)
            restart_action.triggered.connect(self.restart_core)
            tools_menu.addAction(restart_action)
            
            clear_cache_action = QAction("🗑️ پاکسازی حافظه نهان", self)
            clear_cache_action.triggered.connect(self.clear_cache)
            tools_menu.addAction(clear_cache_action)
            
            tools_menu.addSeparator()
            
            settings_action = QAction("⚙️ تنظیمات سیستم...", self)
            settings_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(7))
            tools_menu.addAction(settings_action)
            
            help_menu = menubar.addMenu("❓ کمک")
            
            docs_action = QAction("📚 مستندات", self)
            docs_action.triggered.connect(self.show_documentation)
            help_menu.addAction(docs_action)
            
            about_action = QAction("ℹ️ درباره ImanCore", self)
            about_action.triggered.connect(self.show_about_dialog)
            help_menu.addAction(about_action)
        
        def _create_dashboard_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            stats_layout = QHBoxLayout()
            
            self.stat_cards = {
                'api_keys': self._create_stat_card("🔑 API Keys", "0", "کلید فعال", "#3498db"),
                'files': self._create_stat_card("📁 فایل‌ها", "0", "فایل آپلود شده", "#2ecc71"),
                'learning': self._create_stat_card("🧠 یادگیری", "0", "رکورد آموزشی", "#9b59b6"),
                'models': self._create_stat_card("🤖 مدل‌ها", "0", "مدل فعال", "#e74c3c"),
                'activities': self._create_stat_card("📋 فعالیت‌ها", "0", "لاگ امروز", "#f39c12")
            }
            
            for card in self.stat_cards.values():
                stats_layout.addWidget(card)
            
            layout.addLayout(stats_layout)
            
            splitter = QSplitter(Qt.Horizontal)
            
            recent_group = QGroupBox("📊 فعالیت اخیر")
            recent_layout = QVBoxLayout()
            
            self.recent_table = QTableWidget()
            self.recent_table.setColumnCount(4)
            self.recent_table.setHorizontalHeaderLabels(["زمان", "عملیات", "جزئیات", "وضعیت"])
            self.recent_table.horizontalHeader().setStretchLastSection(True)
            self.recent_table.setAlternatingRowColors(True)
            self.recent_table.setMaximumHeight(300)
            
            recent_layout.addWidget(self.recent_table)
            recent_group.setLayout(recent_layout)
            
            chart_group = QGroupBox("📈 نمودار سریع")
            chart_layout = QVBoxLayout()
            
            chart_widget = QWidget()
            chart_widget.setFixedHeight(250)
            chart_widget.setStyleSheet("""
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #dee2e6;
            """)
            
            chart_label = QLabel("فعالیت سیستم در ۷ روز گذشته")
            chart_label.setAlignment(Qt.AlignCenter)
            chart_label.setStyleSheet("font-weight: bold; padding: 10px;")
            
            chart_layout.addWidget(chart_label)
            chart_layout.addWidget(chart_widget)
            chart_group.setLayout(chart_layout)
            
            splitter.addWidget(recent_group)
            splitter.addWidget(chart_group)
            splitter.setSizes([600, 400])
            
            layout.addWidget(splitter, 1)
            
            quick_actions = QGroupBox("🚀 اقدامات سریع")
            quick_layout = QHBoxLayout()
            
            actions = [
                ("➕ ایجاد API Key", self.show_create_api_dialog, "#3498db"),
                ("📤 آپلود فایل", self.show_upload_dialog, "#2ecc71"),
                ("🤖 آموزش مدل", self.show_train_dl_dialog, "#9b59b6"),
                ("📊 خروجی داده", self.show_export_dialog, "#e74c3c"),
                ("💾 پشتیبان‌گیری", self.show_backup_dialog, "#f39c12"),
                ("⚙️ تنظیمات", lambda: self.tab_widget.setCurrentIndex(7), "#34495e")
            ]
            
            for text, callback, color in actions:
                btn = QPushButton(text)
                btn.clicked.connect(callback)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color};
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }}
                    QPushButton:hover {{
                        background-color: {self._darken_color(color)};
                    }}
                """)
                quick_layout.addWidget(btn)
            
            quick_actions.setLayout(quick_layout)
            layout.addWidget(quick_actions)
            
            return tab
        
        def _create_api_keys_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            toolbar = QHBoxLayout()
            
            self.create_key_btn = QPushButton("➕ ایجاد جدید")
            self.create_key_btn.clicked.connect(self.show_create_api_dialog)
            
            self.revoke_key_btn = QPushButton("🚫 غیرفعال کردن")
            self.revoke_key_btn.clicked.connect(self.revoke_selected_key)
            self.revoke_key_btn.setEnabled(False)
            
            self.delete_key_btn = QPushButton("🗑️ حذف")
            self.delete_key_btn.clicked.connect(self.delete_selected_key)
            self.delete_key_btn.setEnabled(False)
            
            self.copy_key_btn = QPushButton("📋 کپی API Key")
            self.copy_key_btn.clicked.connect(self.copy_selected_key)
            self.copy_key_btn.setEnabled(False)
            
            self.refresh_keys_btn = QPushButton("🔄 به‌روزرسانی")
            self.refresh_keys_btn.clicked.connect(self.refresh_api_keys)
            
            toolbar.addWidget(self.create_key_btn)
            toolbar.addWidget(self.revoke_key_btn)
            toolbar.addWidget(self.delete_key_btn)
            toolbar.addWidget(self.copy_key_btn)
            toolbar.addStretch()
            toolbar.addWidget(self.refresh_keys_btn)
            
            layout.addLayout(toolbar)
            
            self.api_keys_table = QTableWidget()
            self.api_keys_table.setColumnCount(8)
            self.api_keys_table.setHorizontalHeaderLabels([
                "نام اپ", "مالک", "ایجاد شده", "آخرین استفاده", "درخواست‌ها", "محدودیت", "وضعیت", "مجوزها"
            ])
            self.api_keys_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.api_keys_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.api_keys_table.horizontalHeader().setStretchLastSection(True)
            self.api_keys_table.setAlternatingRowColors(True)
            self.api_keys_table.itemSelectionChanged.connect(self.on_api_key_selected)
            
            layout.addWidget(self.api_keys_table, 1)
            
            details_group = QGroupBox("جزئیات API Key")
            details_layout = QVBoxLayout()
            
            self.key_details = QTextEdit()
            self.key_details.setReadOnly(True)
            self.key_details.setMaximumHeight(150)
            self.key_details.setStyleSheet("""
                QTextEdit {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            
            details_layout.addWidget(self.key_details)
            details_group.setLayout(details_layout)
            layout.addWidget(details_group)
            
            return tab
        
        def _create_files_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            toolbar = QHBoxLayout()
            
            self.upload_file_btn = QPushButton("📤 آپلود فایل")
            self.upload_file_btn.clicked.connect(self.show_upload_dialog)
            
            self.download_file_btn = QPushButton("📥 دانلود")
            self.download_file_btn.clicked.connect(self.download_selected_file)
            self.download_file_btn.setEnabled(False)
            
            self.view_file_btn = QPushButton("👁️ مشاهده")
            self.view_file_btn.clicked.connect(self.view_selected_file)
            self.view_file_btn.setEnabled(False)
            
            self.delete_file_btn = QPushButton("🗑️ حذف")
            self.delete_file_btn.clicked.connect(self.delete_selected_file)
            self.delete_file_btn.setEnabled(False)
            
            self.process_file_btn = QPushButton("⚙️ پردازش")
            self.process_file_btn.clicked.connect(self.process_selected_file)
            self.process_file_btn.setEnabled(False)
            
            self.refresh_files_btn = QPushButton("🔄 به‌روزرسانی")
            self.refresh_files_btn.clicked.connect(self.refresh_files)
            
            toolbar.addWidget(self.upload_file_btn)
            toolbar.addWidget(self.download_file_btn)
            toolbar.addWidget(self.view_file_btn)
            toolbar.addWidget(self.delete_file_btn)
            toolbar.addWidget(self.process_file_btn)
            toolbar.addStretch()
            toolbar.addWidget(self.refresh_files_btn)
            
            layout.addLayout(toolbar)
            
            self.files_table = QTableWidget()
            self.files_table.setColumnCount(8)
            self.files_table.setHorizontalHeaderLabels([
                "نام فایل", "نوع", "حجم", "آپلود شده", "تاریخ", "پردازش شده", "هش", "مسیر"
            ])
            self.files_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.files_table.setSelectionMode(QAbstractItemView.SingleSelection)
            self.files_table.horizontalHeader().setStretchLastSection(True)
            self.files_table.setAlternatingRowColors(True)
            self.files_table.itemSelectionChanged.connect(self.on_file_selected)
            
            layout.addWidget(self.files_table, 1)
            
            preview_group = QGroupBox("پیش‌نمایش فایل")
            preview_layout = QVBoxLayout()
            
            self.file_preview = QTextEdit()
            self.file_preview.setReadOnly(True)
            self.file_preview.setMaximumHeight(200)
            self.file_preview.setStyleSheet("""
                QTextEdit {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            
            preview_layout.addWidget(self.file_preview)
            preview_group.setLayout(preview_layout)
            layout.addWidget(preview_group)
            
            return tab
        
        def _create_learning_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            controls = QHBoxLayout()
            
            self.learning_status = QLabel("🟢 یادگیری فعال")
            self.learning_status.setStyleSheet("color: green; font-weight: bold;")
            
            self.pause_learning_btn = QPushButton("⏸️ توقف")
            self.pause_learning_btn.clicked.connect(self.toggle_learning)
            
            self.add_learning_btn = QPushButton("➕ افزودن داده")
            self.add_learning_btn.clicked.connect(self.show_add_learning_dialog)
            
            self.clear_learning_btn = QPushButton("🗑️ پاکسازی")
            self.clear_learning_btn.clicked.connect(self.clear_learning_data)
            
            self.train_now_btn = QPushButton("🎓 آموزش اکنون")
            self.train_now_btn.clicked.connect(self.train_models_now)
            
            controls.addWidget(self.learning_status)
            controls.addWidget(self.pause_learning_btn)
            controls.addWidget(self.add_learning_btn)
            controls.addWidget(self.clear_learning_btn)
            controls.addWidget(self.train_now_btn)
            controls.addStretch()
            
            layout.addLayout(controls)
            
            self.learning_table = QTableWidget()
            self.learning_table.setColumnCount(7)
            self.learning_table.setHorizontalHeaderLabels([
                "اپلیکیشن", "نوع داده", "محتوا", "برچسب‌ها", "ویژگی‌ها", "زمان", "پردازش شده"
            ])
            self.learning_table.horizontalHeader().setStretchLastSection(True)
            self.learning_table.setAlternatingRowColors(True)
            
            layout.addWidget(self.learning_table, 1)
            
            chart_group = QGroupBox("📈 روند یادگیری")
            chart_layout = QVBoxLayout()
            
            chart_widget = QWidget()
            chart_widget.setFixedHeight(150)
            chart_widget.setStyleSheet("""
                background-color: #e8f4fd;
                border-radius: 10px;
                border: 1px solid #b6d7e8;
            """)
            
            chart_layout.addWidget(chart_widget)
            chart_group.setLayout(chart_layout)
            layout.addWidget(chart_group)
            
            return tab
        
        def _create_deep_learning_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            train_group = QGroupBox("🎯 آموزش مدل جدید")
            train_layout = QFormLayout()
            
            self.dl_app_name = QLineEdit()
            self.dl_app_name.setPlaceholderText("نام اپلیکیشن")
            
            self.dl_data_type = QComboBox()
            self.dl_data_type.addItems(["text", "numeric", "image", "custom"])
            
            self.dl_epochs = QSpinBox()
            self.dl_epochs.setRange(1, 10000)
            self.dl_epochs.setValue(100)
            
            self.dl_layers = QLineEdit()
            self.dl_layers.setPlaceholderText("مثلاً: 64,32,16")
            self.dl_layers.setText("64,32,16")
            
            self.dl_activation = QComboBox()
            self.dl_activation.addItems(["relu", "sigmoid", "tanh", "leaky_relu"])
            
            self.dl_output_activation = QComboBox()
            self.dl_output_activation.addItems(["softmax", "sigmoid", "linear"])
            
            train_layout.addRow("نام اپلیکیشن:", self.dl_app_name)
            train_layout.addRow("نوع داده:", self.dl_data_type)
            train_layout.addRow("تعداد دوره‌ها:", self.dl_epochs)
            train_layout.addRow("لایه‌های مخفی:", self.dl_layers)
            train_layout.addRow("تابع فعال‌سازی:", self.dl_activation)
            train_layout.addRow("تابع خروجی:", self.dl_output_activation)
            
            train_group.setLayout(train_layout)
            layout.addWidget(train_group)
            
            data_group = QGroupBox("📊 داده آموزشی")
            data_layout = QVBoxLayout()
            
            self.training_data = QTextEdit()
            self.training_data.setPlaceholderText("ورودی‌ها و خروجی‌ها به فرمت JSON...")
            self.training_data.setMaximumHeight(120)
            
            data_buttons = QHBoxLayout()
            
            self.load_data_btn = QPushButton("📂 بارگذاری فایل")
            self.load_data_btn.clicked.connect(self.load_training_data)
            
            self.generate_data_btn = QPushButton("🎲 ایجاد داده نمونه")
            self.generate_data_btn.clicked.connect(self.generate_sample_data)
            
            data_buttons.addWidget(self.load_data_btn)
            data_buttons.addWidget(self.generate_data_btn)
            data_buttons.addStretch()
            
            data_layout.addWidget(self.training_data)
            data_layout.addLayout(data_buttons)
            data_group.setLayout(data_layout)
            layout.addWidget(data_group)
            
            train_buttons = QHBoxLayout()
            
            self.start_training_btn = QPushButton("🚀 شروع آموزش")
            self.start_training_btn.clicked.connect(self.start_dl_training)
            self.start_training_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
            
            self.cancel_training_btn = QPushButton("❌ لغو")
            self.cancel_training_btn.clicked.connect(self.cancel_training)
            self.cancel_training_btn.setEnabled(False)
            
            self.load_model_btn = QPushButton("📂 بارگذاری مدل")
            self.load_model_btn.clicked.connect(self.load_dl_model)
            
            self.save_model_btn = QPushButton("💾 ذخیره مدل")
            self.save_model_btn.clicked.connect(self.save_dl_model)
            
            train_buttons.addWidget(self.start_training_btn)
            train_buttons.addWidget(self.cancel_training_btn)
            train_buttons.addWidget(self.load_model_btn)
            train_buttons.addWidget(self.save_model_btn)
            train_buttons.addStretch()
            
            layout.addLayout(train_buttons)
            
            models_group = QGroupBox("🤖 مدل‌های فعال")
            models_layout = QVBoxLayout()
            
            self.dl_models_table = QTableWidget()
            self.dl_models_table.setColumnCount(6)
            self.dl_models_table.setHorizontalHeaderLabels([
                "نام مدل", "معماری", "پارامترها", "دقت", "ایجاد شده", "وضعیت"
            ])
            self.dl_models_table.horizontalHeader().setStretchLastSection(True)
            self.dl_models_table.setAlternatingRowColors(True)
            
            models_layout.addWidget(self.dl_models_table)
            models_group.setLayout(models_layout)
            layout.addWidget(models_group, 1)
            
            return tab
        
        def _create_stats_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            stats_grid = QGridLayout()
            stats_grid.setSpacing(10)
            
            api_stats = QGroupBox("📊 آمار API")
            api_stats_layout = QVBoxLayout()
            
            self.api_stats_text = QTextEdit()
            self.api_stats_text.setReadOnly(True)
            self.api_stats_text.setStyleSheet("""
                QTextEdit {
                    background-color: #e3f2fd;
                    border: 1px solid #90caf9;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            api_stats_layout.addWidget(self.api_stats_text)
            api_stats.setLayout(api_stats_layout)
            stats_grid.addWidget(api_stats, 0, 0)
            
            file_stats = QGroupBox("📁 آمار فایل‌ها")
            file_stats_layout = QVBoxLayout()
            
            self.file_stats_text = QTextEdit()
            self.file_stats_text.setReadOnly(True)
            self.file_stats_text.setStyleSheet("""
                QTextEdit {
                    background-color: #e8f5e9;
                    border: 1px solid #a5d6a7;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            file_stats_layout.addWidget(self.file_stats_text)
            file_stats.setLayout(file_stats_layout)
            stats_grid.addWidget(file_stats, 0, 1)
            
            learning_stats = QGroupBox("🧠 آمار یادگیری")
            learning_stats_layout = QVBoxLayout()
            
            self.learning_stats_text = QTextEdit()
            self.learning_stats_text.setReadOnly(True)
            self.learning_stats_text.setStyleSheet("""
                QTextEdit {
                    background-color: #f3e5f5;
                    border: 1px solid #ce93d8;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            learning_stats_layout.addWidget(self.learning_stats_text)
            learning_stats.setLayout(learning_stats_layout)
            stats_grid.addWidget(learning_stats, 1, 0)
            
            system_stats = QGroupBox("⚙️ آمار سیستم")
            system_stats_layout = QVBoxLayout()
            
            self.system_stats_text = QTextEdit()
            self.system_stats_text.setReadOnly(True)
            self.system_stats_text.setStyleSheet("""
                QTextEdit {
                    background-color: #fff3e0;
                    border: 1px solid #ffcc80;
                    border-radius: 5px;
                    padding: 10px;
                    font-family: 'Courier New', monospace;
                }
            """)
            system_stats_layout.addWidget(self.system_stats_text)
            system_stats.setLayout(system_stats_layout)
            stats_grid.addWidget(system_stats, 1, 1)
            
            layout.addLayout(stats_grid, 1)
            
            button_layout = QHBoxLayout()
            
            self.export_stats_btn = QPushButton("📊 خروجی Excel")
            self.export_stats_btn.clicked.connect(self.export_stats)
            
            self.refresh_stats_btn = QPushButton("🔄 به‌روزرسانی")
            self.refresh_stats_btn.clicked.connect(self.refresh_stats)
            
            self.save_stats_btn = QPushButton("💾 ذخیره آمار")
            self.save_stats_btn.clicked.connect(self.save_stats)
            
            button_layout.addWidget(self.export_stats_btn)
            button_layout.addWidget(self.refresh_stats_btn)
            button_layout.addWidget(self.save_stats_btn)
            button_layout.addStretch()
            
            layout.addLayout(button_layout)
            
            return tab
        
        def _create_activity_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            filter_group = QGroupBox("فیلترها")
            filter_layout = QHBoxLayout()
            
            filter_layout.addWidget(QLabel("نوع:"))
            self.activity_type = QComboBox()
            self.activity_type.addItems(["همه", "upload", "learn", "predict", "register", "export", "backup"])
            
            filter_layout.addWidget(self.activity_type)
            
            filter_layout.addWidget(QLabel("وضعیت:"))
            self.activity_status = QComboBox()
            self.activity_status.addItems(["همه", "success", "error", "warning"])
            
            filter_layout.addWidget(self.activity_status)
            
            filter_layout.addWidget(QLabel("از:"))
            self.activity_from = QDateEdit()
            self.activity_from.setDate(QDate.currentDate().addDays(-7))
            
            filter_layout.addWidget(self.activity_from)
            
            filter_layout.addWidget(QLabel("تا:"))
            self.activity_to = QDateEdit()
            self.activity_to.setDate(QDate.currentDate())
            
            filter_layout.addWidget(self.activity_to)
            
            self.apply_filter_btn = QPushButton("🔍 اعمال")
            self.apply_filter_btn.clicked.connect(self.apply_activity_filter)
            
            self.clear_filter_btn = QPushButton("🗑️ پاک کردن")
            self.clear_filter_btn.clicked.connect(self.clear_activity_filter)
            
            filter_layout.addWidget(self.apply_filter_btn)
            filter_layout.addWidget(self.clear_filter_btn)
            filter_layout.addStretch()
            
            filter_group.setLayout(filter_layout)
            layout.addWidget(filter_group)
            
            self.activity_table = QTableWidget()
            self.activity_table.setColumnCount(6)
            self.activity_table.setHorizontalHeaderLabels([
                "زمان", "اپلیکیشن", "عملیات", "جزئیات", "وضعیت", "شناسه"
            ])
            self.activity_table.horizontalHeader().setStretchLastSection(True)
            self.activity_table.setAlternatingRowColors(True)
            self.activity_table.setSortingEnabled(True)
            
            layout.addWidget(self.activity_table, 1)
            
            button_layout = QHBoxLayout()
            
            self.refresh_activity_btn = QPushButton("🔄 به‌روزرسانی")
            self.refresh_activity_btn.clicked.connect(self.refresh_activity)
            
            self.clear_activity_btn = QPushButton("🗑️ پاکسازی")
            self.clear_activity_btn.clicked.connect(self.clear_activity)
            
            self.export_activity_btn = QPushButton("📊 خروجی")
            self.export_activity_btn.clicked.connect(self.export_activity)
            
            button_layout.addWidget(self.refresh_activity_btn)
            button_layout.addWidget(self.clear_activity_btn)
            button_layout.addWidget(self.export_activity_btn)
            button_layout.addStretch()
            
            layout.addLayout(button_layout)
            
            return tab
        
        def _create_settings_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            system_group = QGroupBox("⚙️ تنظیمات سیستم")
            system_layout = QFormLayout()
            
            self.setting_max_file_size = QSpinBox()
            self.setting_max_file_size.setRange(1, 10000)
            self.setting_max_file_size.setValue(100)
            self.setting_max_file_size.setSuffix(" MB")
            
            self.setting_learning_enabled = QCheckBox("فعال")
            self.setting_learning_enabled.setChecked(True)
            
            self.setting_deep_learning_enabled = QCheckBox("فعال")
            self.setting_deep_learning_enabled.setChecked(True)
            
            self.setting_auto_backup = QCheckBox("فعال")
            self.setting_auto_backup.setChecked(True)
            
            self.setting_backup_hours = QSpinBox()
            self.setting_backup_hours.setRange(1, 168)
            self.setting_backup_hours.setValue(24)
            self.setting_backup_hours.setSuffix(" ساعت")
            
            self.setting_log_level = QComboBox()
            self.setting_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
            self.setting_log_level.setCurrentText("INFO")
            
            system_layout.addRow("حداکثر حجم فایل:", self.setting_max_file_size)
            system_layout.addRow("یادگیری فعال:", self.setting_learning_enabled)
            system_layout.addRow("دیپ لرنینگ فعال:", self.setting_deep_learning_enabled)
            system_layout.addRow("پشتیبان‌گیری خودکار:", self.setting_auto_backup)
            system_layout.addRow("فاصله پشتیبان‌گیری:", self.setting_backup_hours)
            system_layout.addRow("سطح لاگ:", self.setting_log_level)
            
            system_group.setLayout(system_layout)
            layout.addWidget(system_group)
            
            storage_group = QGroupBox("💾 تنظیمات ذخیره‌سازی")
            storage_layout = QFormLayout()
            
            self.setting_storage_path = QLineEdit("./imancore_storage")
            
            self.setting_backup_path = QLineEdit("./imancore_backups")
            
            self.setting_max_storage = QSpinBox()
            self.setting_max_storage.setRange(100, 100000)
            self.setting_max_storage.setValue(10000)
            self.setting_max_storage.setSuffix(" MB")
            
            storage_layout.addRow("مسیر ذخیره‌سازی:", self.setting_storage_path)
            storage_layout.addRow("مسیر پشتیبان:", self.setting_backup_path)
            storage_layout.addRow("حداکثر فضای ذخیره‌سازی:", self.setting_max_storage)
            
            storage_group.setLayout(storage_layout)
            layout.addWidget(storage_group)
            
            api_group = QGroupBox("🌐 تنظیمات API")
            api_layout = QFormLayout()
            
            self.setting_api_rate_limit = QSpinBox()
            self.setting_api_rate_limit.setRange(1, 100000)
            self.setting_api_rate_limit.setValue(1000)
            self.setting_api_rate_limit.setSuffix(" درخواست در ساعت")
            
            self.setting_api_timeout = QSpinBox()
            self.setting_api_timeout.setRange(1, 300)
            self.setting_api_timeout.setValue(30)
            self.setting_api_timeout.setSuffix(" ثانیه")
            
            api_layout.addRow("محدودیت نرخ API:", self.setting_api_rate_limit)
            api_layout.addRow("زمان‌سنج API:", self.setting_api_timeout)
            
            api_group.setLayout(api_layout)
            layout.addWidget(api_group)
            
            button_layout = QHBoxLayout()
            
            self.save_settings_btn = QPushButton("💾 ذخیره تنظیمات")
            self.save_settings_btn.clicked.connect(self.save_settings)
            self.save_settings_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
            """)
            
            self.load_settings_btn = QPushButton("📂 بارگذاری تنظیمات")
            self.load_settings_btn.clicked.connect(self.load_settings)
            
            self.reset_settings_btn = QPushButton("🔄 تنظیمات پیش‌فرض")
            self.reset_settings_btn.clicked.connect(self.reset_settings)
            
            button_layout.addWidget(self.save_settings_btn)
            button_layout.addWidget(self.load_settings_btn)
            button_layout.addWidget(self.reset_settings_btn)
            button_layout.addStretch()
            
            layout.addLayout(button_layout)
            layout.addStretch()
            
            return tab
        
        def _create_stat_card(self, title: str, value: str, subtitle: str, color: str) -> QGroupBox:
            card = QGroupBox(title)
            card.setStyleSheet(f"""
                QGroupBox {{
                    border: 2px solid {color};
                    border-radius: 10px;
                    padding: 15px;
                    background-color: white;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: {color};
                    font-weight: bold;
                }}
            """)
            
            layout = QVBoxLayout(card)
            layout.setSpacing(5)
            
            value_label = QLabel(value)
            value_label.setStyleSheet("""
                font-size: 24px; 
                font-weight: bold; 
                color: #333;
            """)
            value_label.setAlignment(Qt.AlignCenter)
            
            subtitle_label = QLabel(subtitle)
            subtitle_label.setStyleSheet("""
                font-size: 12px; 
                color: #777;
            """)
            subtitle_label.setAlignment(Qt.AlignCenter)
            
            layout.addWidget(value_label)
            layout.addWidget(subtitle_label)
            
            return card
        
        def _darken_color(self, color: str) -> str:
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                
                r = max(0, r - 30)
                g = max(0, g - 30)
                b = max(0, b - 30)
                
                return f'#{r:02x}{g:02x}{b:02x}'
            
            return color
        
        def _load_initial_data(self):
            self.refresh_api_keys()
            self.refresh_files()
            self.refresh_activity()
            self.refresh_stats()
            self.load_settings()
        
        # توابع API Keys
        def show_create_api_dialog(self):
            dialog = CreateApiKeyDialog(self.core, self)
            if dialog.exec():
                self.refresh_api_keys()
        
        def refresh_api_keys(self):
            try:
                api_keys = self.core.api_manager.list_api_keys()
                
                self.api_keys_table.setRowCount(len(api_keys))
                
                for i, key_info in enumerate(api_keys):
                    status = "🟢 فعال" if key_info.is_active else "🔴 غیرفعال"
                    status_color = "green" if key_info.is_active else "red"
                    
                    items = [
                        QTableWidgetItem(key_info.app_name),
                        QTableWidgetItem(key_info.owner),
                        QTableWidgetItem(key_info.created_at[:19]),
                        QTableWidgetItem(key_info.last_used[:19]),
                        QTableWidgetItem(str(key_info.total_requests)),
                        QTableWidgetItem(str(key_info.rate_limit)),
                        QTableWidgetItem(status),
                        QTableWidgetItem(", ".join(key_info.permissions))
                    ]
                    
                    items[6].setForeground(QColor(status_color))
                    
                    for j, item in enumerate(items):
                        self.api_keys_table.setItem(i, j, item)
                    
                    items[0].setData(Qt.UserRole, key_info.key_id)
                
                self.api_keys_table.resizeColumnsToContents()
                self.stat_cards['api_keys'].findChild(QLabel).setText(str(len(api_keys)))
                
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در بارگذاری API Keys: {str(e)}")
        
        def on_api_key_selected(self):
            selected_items = self.api_keys_table.selectedItems()
            if selected_items:
                key_id = selected_items[0].data(Qt.UserRole)
                
                api_keys = self.core.api_manager.list_api_keys()
                key_info = next((k for k in api_keys if k.key_id == key_id), None)
                
                if key_info:
                    details = f"""🔑 API Key Information:
                    
📱 Application: {key_info.app_name}
👤 Owner: {key_info.owner}
📅 Created: {key_info.created_at}
⏰ Last Used: {key_info.last_used}
📊 Total Requests: {key_info.total_requests}
🚦 Rate Limit: {key_info.rate_limit}/hour
✅ Status: {'Active' if key_info.is_active else 'Inactive'}
📋 Permissions: {', '.join(key_info.permissions)}
🔑 Key ID: {key_info.key_id}"""
                    
                    self.key_details.setText(details)
                    
                    self.revoke_key_btn.setEnabled(True)
                    self.delete_key_btn.setEnabled(True)
                    self.copy_key_btn.setEnabled(True)
                else:
                    self.key_details.setText("مشکل در بارگذاری اطلاعات")
                    self.revoke_key_btn.setEnabled(False)
                    self.delete_key_btn.setEnabled(False)
                    self.copy_key_btn.setEnabled(False)
            else:
                self.key_details.setText("هیچ API Key انتخاب نشده است")
                self.revoke_key_btn.setEnabled(False)
                self.delete_key_btn.setEnabled(False)
                self.copy_key_btn.setEnabled(False)
        
        def revoke_selected_key(self):
            selected_items = self.api_keys_table.selectedItems()
            if not selected_items:
                return
            
            key_id = selected_items[0].data(Qt.UserRole)
            
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید این API Key را غیرفعال کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                api_keys = self.core.api_manager.list_api_keys()
                key_info = next((k for k in api_keys if k.key_id == key_id), None)
                
                if key_info:
                    success = self.core.api_manager.update_api_key(
                        key_id, is_active=not key_info.is_active
                    )
                    
                    if success:
                        QMessageBox.information(
                            self, "موفقیت",
                            f"API Key {'غیرفعال' if key_info.is_active else 'فعال'} شد"
                        )
                        self.refresh_api_keys()
                    else:
                        QMessageBox.critical(self, "خطا", "خطا در به‌روزرسانی API Key")
        
        def delete_selected_key(self):
            selected_items = self.api_keys_table.selectedItems()
            if not selected_items:
                return
            
            key_id = selected_items[0].data(Qt.UserRole)
            
            reply = QMessageBox.warning(
                self, "هشدار",
                "آیا مطمئن هستید که می‌خواهید این API Key را حذف کنید؟\nاین عمل قابل بازگشت نیست!",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                success = self.core.api_manager.delete_api_key(key_id)
                
                if success:
                    QMessageBox.information(self, "موفقیت", "API Key حذف شد")
                    self.refresh_api_keys()
                else:
                    QMessageBox.critical(self, "خطا", "خطا در حذف API Key")
        
        def copy_selected_key(self):
            selected_items = self.api_keys_table.selectedItems()
            if not selected_items:
                return
            
            key_id = selected_items[0].data(Qt.UserRole)
            
            api_keys = self.core.api_manager.list_api_keys()
            key_info = next((k for k in api_keys if k.key_id == key_id), None)
            
            if key_info:
                QMessageBox.information(
                    self, "اطلاعات",
                    "API Key اصلی در دیتابیس ذخیره شده است.\n"
                    f"برای دسترسی با شناسه {key_id} درخواست دهید."
                )
        
        # توابع فایل‌ها
        def show_upload_dialog(self):
            api_keys = self.core.api_manager.list_api_keys()
            if not api_keys:
                QMessageBox.warning(
                    self, "هشدار",
                    "برای آپلود فایل نیاز به API Key دارید.\nلطفاً ابتدا یک API Key ایجاد کنید."
                )
                return
            
            active_keys = [k for k in api_keys if k.is_active]
            if not active_keys:
                QMessageBox.warning(self, "هشدار", "هیچ API Key فعالی پیدا نشد")
                return
            
            api_key_info = active_keys[0]
            
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                
                progress_dialog = LoadingDialog("در حال آپلود فایل‌ها...", self)
                
                self.upload_worker = BackgroundWorker(
                    task_type='process_files',
                    core=self.core,
                    api_key=api_key_info.key_id,
                    file_paths=selected_files
                )
                
                self.upload_worker.progress.connect(progress_dialog.update_progress)
                self.upload_worker.message.connect(lambda msg: progress_dialog.update_progress(
                    progress_dialog.progress_bar.value(), msg
                ))
                self.upload_worker.finished.connect(
                    lambda result: self._on_upload_finished(result, progress_dialog)
                )
                
                progress_dialog.show()
                self.upload_worker.start()
        
        def _on_upload_finished(self, result: Dict, progress_dialog: LoadingDialog):
            progress_dialog.close()
            
            if result.get('success'):
                successful = result.get('successful_uploads', 0)
                total = result.get('total_files', 0)
                
                msg = f"""✅ آپلود تکمیل شد!

📊 نتایج:
• فایل‌های ارسال شده: {total}
• آپلود موفق: {successful}
• ناموفق: {total - successful}"""
                
                QMessageBox.information(self, "آپلود تکمیل شد", msg)
                self.refresh_files()
            else:
                QMessageBox.critical(self, "خطا", 
                                   f"خطا در آپلود: {result.get('error', 'خطای ناشناخته')}")
        
        def refresh_files(self):
            try:
                api_keys = self.core.api_manager.list_api_keys()
                if not api_keys:
                    return
                
                active_keys = [k for k in api_keys if k.is_active]
                if not active_keys:
                    return
                
                api_key_info = active_keys[0]
                files = self.core.query_files(api_key_info.key_id)
                
                if files and 'error' in files[0]:
                    return
                
                self.files_table.setRowCount(len(files))
                
                for i, file_info in enumerate(files):
                    size_mb = file_info['size_bytes'] / (1024 * 1024)
                    size_str = f"{size_mb:.2f} MB"
                    
                    processed = "✅ بله" if file_info.get('processed', False) else "⏳ خیر"
                    
                    items = [
                        QTableWidgetItem(file_info['original_name']),
                        QTableWidgetItem(file_info['file_type']),
                        QTableWidgetItem(size_str),
                        QTableWidgetItem(file_info.get('uploaded_by', 'Unknown')),
                        QTableWidgetItem(file_info['upload_time'][:19]),
                        QTableWidgetItem(processed),
                        QTableWidgetItem(file_info['hash_md5'][:8] + "..."),
                        QTableWidgetItem(file_info.get('storage_path', 'Unknown'))
                    ]
                    
                    for j, item in enumerate(items):
                        self.files_table.setItem(i, j, item)
                    
                    items[0].setData(Qt.UserRole, file_info['file_id'])
                
                self.files_table.resizeColumnsToContents()
                self.stat_cards['files'].findChild(QLabel).setText(str(len(files)))
                
            except Exception as e:
                print(f"Error refreshing files: {e}")
        
        def on_file_selected(self):
            selected_items = self.files_table.selectedItems()
            if selected_items:
                self.download_file_btn.setEnabled(True)
                self.view_file_btn.setEnabled(True)
                self.delete_file_btn.setEnabled(True)
                self.process_file_btn.setEnabled(True)
            else:
                self.download_file_btn.setEnabled(False)
                self.view_file_btn.setEnabled(False)
                self.delete_file_btn.setEnabled(False)
                self.process_file_btn.setEnabled(False)
                
                self.file_preview.clear()
        
        def view_selected_file(self):
            selected_items = self.files_table.selectedItems()
            if not selected_items:
                return
            
            file_id = selected_items[0].data(Qt.UserRole)
            
            api_keys = self.core.api_manager.list_api_keys()
            if not api_keys:
                return
            
            active_keys = [k for k in api_keys if k.is_active]
            if not active_keys:
                return
            
            api_key_info = active_keys[0]
            
            try:
                result = self.core.get_file_content(api_key_info.key_id, file_id)
                
                if 'success' in result and result['success']:
                    content = result.get('content', '')
                    filename = result.get('filename', 'Unknown')
                    
                    self.file_preview.setText(f"""📁 فایل: {filename}

{content}

{'⚠️ فقط بخشی از فایل نمایش داده شد' if result.get('full_content_available', False) else '✅ تمام محتوا نمایش داده شد'}""")
                else:
                    self.file_preview.setText(f"❌ خطا: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                self.file_preview.setText(f"❌ خطا در بارگذاری فایل: {str(e)}")
        
        def download_selected_file(self):
            QMessageBox.information(
                self, "اطلاعات",
                "برای دانلود فایل از API استفاده کنید:\nGET /files/{file_id}"
            )
        
        def delete_selected_file(self):
            QMessageBox.information(
                self, "اطلاعات",
                "این قابلیت در نسخه فعلی پیاده‌سازی نشده است."
            )
        
        def process_selected_file(self):
            QMessageBox.information(
                self, "اطلاعات",
                "این قابلیت در نسخه فعلی پیاده‌سازی نشده است."
            )
        
        # توابع یادگیری
        def show_add_learning_dialog(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای افزودن داده آموزشی استفاده کنید:\nPOST /learn"
            )
        
        def toggle_learning(self):
            current_text = self.pause_learning_btn.text()
            if current_text == "⏸️ توقف":
                self.pause_learning_btn.setText("▶️ شروع")
                self.learning_status.setText("🔴 یادگیری متوقف")
                self.learning_status.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.pause_learning_btn.setText("⏸️ توقف")
                self.learning_status.setText("🟢 یادگیری فعال")
                self.learning_status.setStyleSheet("color: green; font-weight: bold;")
        
        def clear_learning_data(self):
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید تمام داده‌های یادگیری را پاک کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                QMessageBox.information(
                    self, "اطلاعات",
                    "این قابلیت در نسخه فعلی پیاده‌سازی نشده است."
                )
        
        def train_models_now(self):
            QMessageBox.information(
                self, "اطلاعات",
                "لطفاً از تب دیپ لرنینگ برای آموزش مدل‌ها استفاده کنید."
            )
        
        # توابع دیپ لرنینگ
        def show_train_dl_dialog(self):
            self.tab_widget.setCurrentIndex(4)
        
        def load_training_data(self):
            file_dialog = QFileDialog()
            file_dialog.setNameFilter("JSON Files (*.json);;CSV Files (*.csv);;All Files (*.*)")
            
            if file_dialog.exec():
                selected_files = file_dialog.selectedFiles()
                if selected_files:
                    try:
                        with open(selected_files[0], 'r', encoding='utf-8') as f:
                            content = f.read()
                            self.training_data.setText(content)
                    except Exception as e:
                        QMessageBox.critical(self, "خطا", f"خطا در بارگذاری فایل: {str(e)}")
        
        def generate_sample_data(self):
            sample_data = {
                "X": [
                    [0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8, 0.9, 1.0],
                    [0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9, 1.0, 0.1],
                    [0.3, 0.4, 0.5, 0.6, 0.7],
                    [0.8, 0.9, 1.0, 0.1, 0.2],
                    [0.4, 0.5, 0.6, 0.7, 0.8],
                    [0.9, 1.0, 0.1, 0.2, 0.3],
                    [0.5, 0.6, 0.7, 0.8, 0.9],
                    [1.0, 0.1, 0.2, 0.3, 0.4]
                ],
                "y": [
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1],
                    [0, 0, 1]
                ]
            }
            
            self.training_data.setText(json.dumps(sample_data, indent=2))
            QMessageBox.information(self, "اطلاعات", "داده نمونه ایجاد شد")
        
        def start_dl_training(self):
            app_name = self.dl_app_name.text().strip()
            if not app_name:
                QMessageBox.warning(self, "هشدار", "لطفاً نام اپلیکیشن را وارد کنید")
                return
            
            try:
                data_text = self.training_data.toPlainText().strip()
                if not data_text:
                    QMessageBox.warning(self, "هشدار", "لطفاً داده آموزشی را وارد کنید")
                    return
                
                try:
                    data = json.loads(data_text)
                    X = data.get('X', [])
                    y = data.get('y', [])
                    
                    if not X or not y:
                        QMessageBox.warning(self, "هشدار", "داده آموزشی باید شامل X و y باشد")
                        return
                    
                    if len(X) != len(y):
                        QMessageBox.warning(self, "هشدار", "تعداد نمونه‌ها در X و y باید برابر باشد")
                        return
                    
                except json.JSONDecodeError:
                    QMessageBox.warning(self, "هشدار", "داده آموزشی باید فرمت JSON معتبر داشته باشد")
                    return
                
                data_type = self.dl_data_type.currentText()
                epochs = self.dl_epochs.value()
                
                layers_text = self.dl_layers.text().strip()
                try:
                    layers = [int(x.strip()) for x in layers_text.split(',') if x.strip()]
                    if not layers:
                        layers = [64, 32, 16]
                except:
                    layers = [64, 32, 16]
                
                api_keys = self.core.api_manager.list_api_keys()
                if not api_keys:
                    QMessageBox.warning(self, "هشدار", "نیاز به API Key دارید")
                    return
                
                active_keys = [k for k in api_keys if k.is_active]
                if not active_keys:
                    QMessageBox.warning(self, "هشدار", "هیچ API Key فعالی پیدا نشد")
                    return
                
                api_key_info = active_keys[0]
                model_key = f"{app_name}_{data_type}_{int(time.time())}"
                
                progress_dialog = LoadingDialog("در حال آموزش مدل...", self)
                
                self.training_worker = BackgroundWorker(
                    task_type='train_model',
                    core=self.core,
                    api_key=api_key_info.key_id,
                    model_key=model_key,
                    X=X,
                    y=y,
                    epochs=epochs,
                    layers=layers
                )
                
                self.training_worker.progress.connect(progress_dialog.update_progress)
                self.training_worker.message.connect(lambda msg: progress_dialog.update_progress(
                    progress_dialog.progress_bar.value(), msg
                ))
                self.training_worker.finished.connect(
                    lambda result: self._on_training_finished(result, progress_dialog)
                )
                
                self.cancel_training_btn.setEnabled(True)
                progress_dialog.cancel_btn.clicked.connect(self.training_worker.stop)
                
                progress_dialog.show()
                self.training_worker.start()
                
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در شروع آموزش: {str(e)}")
        
        def _on_training_finished(self, result: Dict, progress_dialog: LoadingDialog):
            progress_dialog.close()
            self.cancel_training_btn.setEnabled(False)
            
            if result.get('success'):
                model_key = result.get('model_key', 'Unknown')
                final_loss = result.get('final_loss', 0)
                final_accuracy = result.get('final_accuracy', 0)
                layers = result.get('layers', [])
                
                msg = f"""✅ آموزش مدل تکمیل شد!

🔑 Model Key: {model_key}
📊 Final Loss: {final_loss:.4f}
🎯 Final Accuracy: {final_accuracy:.4f}
🏗️ Architecture: {layers}"""
                
                QMessageBox.information(self, "آموزش تکمیل شد", msg)
                self.refresh_stats()
                
            else:
                QMessageBox.critical(
                    self, "خطا", 
                    f"خطا در آموزش مدل: {result.get('error', 'خطای ناشناخته')}"
                )
        
        def cancel_training(self):
            if hasattr(self, 'training_worker') and self.training_worker.isRunning():
                self.training_worker.stop()
                QMessageBox.information(self, "اطلاعات", "آموزش لغو شد")
        
        def load_dl_model(self):
            QMessageBox.information(
                self, "اطلاعات",
                "مدل‌ها به طور خودکار از ذخیره‌سازی بارگذاری می‌شوند."
            )
        
        def save_dl_model(self):
            self.core.learning_engine.save_models()
            QMessageBox.information(self, "موفقیت", "تمام مدل‌ها ذخیره شدند")
        
        # توابع آمار
        def refresh_stats(self):
            try:
                api_keys = self.core.api_manager.list_api_keys()
                if not api_keys:
                    return
                
                active_keys = [k for k in api_keys if k.is_active]
                if not active_keys:
                    return
                
                api_key_info = active_keys[0]
                stats = self.core.get_stats(api_key_info.key_id)
                
                if 'error' in stats:
                    print(f"Error getting stats: {stats['error']}")
                    return
                
                api_stats_text = f"""📊 آمار API

📱 Application: {stats.get('app_name', 'Unknown')}
👤 Owner: {stats.get('owner', 'Unknown')}
📅 Total Requests: {stats.get('total_requests', 0):,}
📋 Permissions: {', '.join(stats.get('permissions', []))}

📈 API Key Stats:
• Total Keys: {stats.get('api_stats', {}).get('total_keys', 0)}
• Active Keys: {stats.get('api_stats', {}).get('active_keys', 0)}
• Total Requests: {stats.get('api_stats', {}).get('total_requests', 0):,}"""
                
                self.api_stats_text.setText(api_stats_text)
                
                file_stats = stats.get('files', {})
                file_stats_text = f"""📁 آمار فایل‌ها

📊 Totals:
• Total Files: {file_stats.get('total', 0):,}
• Total Size: {file_stats.get('total_size_mb', 0):.2f} MB

📂 By Type:"""
                
                for file_type, count in file_stats.get('by_type', {}).items():
                    file_stats_text += f"\n• {file_type}: {count:,}"
                
                self.file_stats_text.setText(file_stats_text)
                
                learning_stats = stats.get('learning', {})
                learning_stats_text = f"""🧠 آمار یادگیری

📊 Totals:
• Total Records: {learning_stats.get('total', 0):,}

📝 By Type:"""
                
                for data_type, count in learning_stats.get('by_type', {}).items():
                    learning_stats_text += f"\n• {data_type}: {count:,}"
                
                self.learning_stats_text.setText(learning_stats_text)
                
                system_stats = stats.get('system', {})
                model_stats = stats.get('models', {})
                system_stats_text = f"""⚙️ آمار سیستم

🖥️ System:
• Version: {system_stats.get('version', 'Unknown')}
• DL Enabled: {'Yes' if system_stats.get('deep_learning_enabled') else 'No'}
• Learning Enabled: {'Yes' if system_stats.get('learning_enabled') else 'No'}
• Storage Used: {stats.get('storage_usage_mb', 0):.2f} MB

🤖 Models:
• Deep Models: {model_stats.get('total_deep_models', 0)}
• Traditional Models: {model_stats.get('total_traditional_models', 0)}
• Total Samples: {model_stats.get('total_samples', 0):,}"""
                
                self.system_stats_text.setText(system_stats_text)
                
                self.stat_cards['files'].findChild(QLabel).setText(
                    str(file_stats.get('total', 0))
                )
                self.stat_cards['learning'].findChild(QLabel).setText(
                    str(learning_stats.get('total', 0))
                )
                self.stat_cards['models'].findChild(QLabel).setText(
                    str(model_stats.get('total_deep_models', 0))
                )
                
            except Exception as e:
                print(f"Error refreshing stats: {e}")
        
        def export_stats(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای خروجی آمار استفاده کنید:\nPOST /export"
            )
        
        def save_stats(self):
            try:
                file_dialog = QFileDialog()
                file_dialog.setAcceptMode(QFileDialog.AcceptSave)
                file_dialog.setNameFilter("JSON Files (*.json)")
                file_dialog.setDefaultSuffix("json")
                
                if file_dialog.exec():
                    selected_files = file_dialog.selectedFiles()
                    if selected_files:
                        api_keys = self.core.api_manager.list_api_keys()
                        if not api_keys:
                            return
                        
                        active_keys = [k for k in api_keys if k.is_active]
                        if not active_keys:
                            return
                        
                        api_key_info = active_keys[0]
                        stats = self.core.get_stats(api_key_info.key_id)
                        
                        with open(selected_files[0], 'w', encoding='utf-8') as f:
                            json.dump(stats, f, indent=2, ensure_ascii=False)
                        
                        QMessageBox.information(self, "موفقیت", "آمار با موفقیت ذخیره شد")
            
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در ذخیره آمار: {str(e)}")
        
        # توابع فعالیت‌ها
        def refresh_activity(self):
            try:
                api_keys = self.core.api_manager.list_api_keys()
                if not api_keys:
                    return
                
                active_keys = [k for k in api_keys if k.is_active]
                if not active_keys:
                    return
                
                api_key_info = active_keys[0]
                activities = self.core.get_activity_log(api_key_info.key_id, limit=50)
                
                if isinstance(activities, list) and activities and 'error' in activities[0]:
                    print(f"Error getting activities: {activities[0]['error']}")
                    return
                
                self.activity_table.setRowCount(len(activities))
                
                today_count = 0
                today = datetime.date.today().isoformat()
                
                for i, activity in enumerate(activities):
                    timestamp = activity.get('timestamp', '')
                    time_str = timestamp[:19] if timestamp else 'Unknown'
                    
                    status = activity.get('status', 'unknown')
                    status_icon = {
                        'success': '✅',
                        'error': '❌',
                        'warning': '⚠️'
                    }.get(status, '📝')
                    
                    items = [
                        QTableWidgetItem(time_str),
                        QTableWidgetItem(activity.get('app_name', 'Unknown')),
                        QTableWidgetItem(activity.get('action', 'Unknown')),
                        QTableWidgetItem(activity.get('details', '')[:100]),
                        QTableWidgetItem(f"{status_icon} {status}"),
                        QTableWidgetItem(activity.get('log_id', 'Unknown')[:8])
                    ]
                    
                    if status == 'success':
                        items[4].setForeground(QColor('green'))
                    elif status == 'error':
                        items[4].setForeground(QColor('red'))
                    elif status == 'warning':
                        items[4].setForeground(QColor('orange'))
                    
                    for j, item in enumerate(items):
                        self.activity_table.setItem(i, j, item)
                    
                    if timestamp.startswith(today):
                        today_count += 1
                
                self.activity_table.resizeColumnsToContents()
                self.stat_cards['activities'].findChild(QLabel).setText(str(today_count))
                
                self.recent_table.setRowCount(min(10, len(activities)))
                for i in range(min(10, len(activities))):
                    activity = activities[i]
                    time_str = activity.get('timestamp', '')[:19]
                    
                    items = [
                        QTableWidgetItem(time_str),
                        QTableWidgetItem(activity.get('action', '')),
                        QTableWidgetItem(activity.get('details', '')[:50]),
                        QTableWidgetItem(activity.get('status', ''))
                    ]
                    
                    for j, item in enumerate(items):
                        self.recent_table.setItem(i, j, item)
                
            except Exception as e:
                print(f"Error refreshing activities: {e}")
        
        def apply_activity_filter(self):
            QMessageBox.information(
                self, "اطلاعات",
                "فیلترها در نسخه فعلی پیاده‌سازی نشده اند."
            )
        
        def clear_activity_filter(self):
            self.activity_type.setCurrentIndex(0)
            self.activity_status.setCurrentIndex(0)
            self.activity_from.setDate(QDate.currentDate().addDays(-7))
            self.activity_to.setDate(QDate.currentDate())
        
        def clear_activity(self):
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید لاگ فعالیت‌ها را پاک کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                QMessageBox.information(
                    self, "اطلاعات",
                    "این قابلیت در نسخه فعلی پیاده‌سازی نشده است."
                )
        
        def export_activity(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای خروجی لاگ استفاده کنید:\nPOST /export"
            )
        
        # توابع تنظیمات
        def load_settings(self):
            try:
                settings = self.core.settings
                
                self.setting_max_file_size.setValue(settings.get('max_file_size_mb', 100))
                self.setting_learning_enabled.setChecked(settings.get('learning_enabled', True))
                self.setting_deep_learning_enabled.setChecked(settings.get('deep_learning_enabled', True))
                self.setting_auto_backup.setChecked(settings.get('auto_backup', True))
                self.setting_backup_hours.setValue(settings.get('auto_backup_hours', 24))
                self.setting_log_level.setCurrentText(settings.get('log_level', 'INFO'))
                
                self.setting_storage_path.setText(settings.get('storage_path', './imancore_storage'))
                self.setting_backup_path.setText(settings.get('backup_path', './imancore_backups'))
                self.setting_max_storage.setValue(settings.get('max_storage_mb', 10000))
                
                self.setting_api_rate_limit.setValue(settings.get('max_api_requests_per_hour', 1000))
                self.setting_api_timeout.setValue(settings.get('api_timeout_seconds', 30))
                
            except Exception as e:
                print(f"Error loading settings: {e}")
        
        def save_settings(self):
            try:
                new_settings = {
                    'max_file_size_mb': self.setting_max_file_size.value(),
                    'learning_enabled': self.setting_learning_enabled.isChecked(),
                    'deep_learning_enabled': self.setting_deep_learning_enabled.isChecked(),
                    'auto_backup': self.setting_auto_backup.isChecked(),
                    'auto_backup_hours': self.setting_backup_hours.value(),
                    'log_level': self.setting_log_level.currentText(),
                    'storage_path': self.setting_storage_path.text(),
                    'backup_path': self.setting_backup_path.text(),
                    'max_storage_mb': self.setting_max_storage.value(),
                    'max_api_requests_per_hour': self.setting_api_rate_limit.value(),
                    'api_timeout_seconds': self.setting_api_timeout.value()
                }
                
                api_keys = self.core.api_manager.list_api_keys()
                admin_keys = [k for k in api_keys if k.owner == 'admin' and k.is_active]
                
                if not admin_keys:
                    QMessageBox.warning(
                        self, "هشدار",
                        "برای تغییر تنظیمات نیاز به API Key مدیر دارید."
                    )
                    return
                
                api_key_info = admin_keys[0]
                result = self.core.update_settings(api_key_info.key_id, new_settings)
                
                if 'error' in result:
                    QMessageBox.critical(self, "خطا", result['error'])
                else:
                    QMessageBox.information(self, "موفقیت", "تنظیمات ذخیره شدند")
                    self.core.settings.update(new_settings)
            
            except Exception as e:
                QMessageBox.critical(self, "خطا", f"خطا در ذخیره تنظیمات: {str(e)}")
        
        def reset_settings(self):
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید تنظیمات را به پیش‌فرض بازنشانی کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                default_settings = {
                    'max_file_size_mb': 100,
                    'learning_enabled': True,
                    'deep_learning_enabled': True,
                    'auto_backup': True,
                    'auto_backup_hours': 24,
                    'log_level': 'INFO',
                    'storage_path': './imancore_storage',
                    'backup_path': './imancore_backups',
                    'max_storage_mb': 10000,
                    'max_api_requests_per_hour': 1000,
                    'api_timeout_seconds': 30
                }
                
                self.core.settings = default_settings
                self.core._save_settings()
                self.load_settings()
                
                QMessageBox.information(self, "موفقیت", "تنظیمات به پیش‌فرض بازنشانی شدند")
        
        # توابع عمومی
        def refresh_all(self):
            self.refresh_api_keys()
            self.refresh_files()
            self.refresh_activity()
            self.refresh_stats()
            
            QMessageBox.information(self, "موفقیت", "همه داده‌ها به‌روزرسانی شدند")
        
        def restart_core(self):
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید هسته را راه‌اندازی مجدد کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.core.shutdown()
                self.core = ImanCore()
                
                QMessageBox.information(self, "موفقیت", "هسته با موفقیت راه‌اندازی مجدد شد")
                self._load_initial_data()
        
        def clear_cache(self):
            reply = QMessageBox.question(
                self, "تأیید",
                "آیا مطمئن هستید که می‌خواهید حافظه نهان را پاک کنید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                temp_dir = Path(self.core.settings['storage_path']) / "temp"
                if temp_dir.exists():
                    try:
                        shutil.rmtree(temp_dir)
                        temp_dir.mkdir(exist_ok=True)
                    except:
                        pass
                
                QMessageBox.information(self, "موفقیت", "حافظه نهان پاکسازی شد")
        
        def show_export_dialog(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای خروجی داده استفاده کنید:\nPOST /export"
            )
        
        def show_backup_dialog(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای پشتیبان‌گیری استفاده کنید:\nPOST /backup"
            )
        
        def show_predict_dialog(self):
            QMessageBox.information(
                self, "اطلاعات",
                "از API برای پیش‌بینی استفاده کنید:\nPOST /predict"
            )
        
        def show_documentation(self):
            QMessageBox.information(
                self, "مستندات",
                """🧠 ImanCore v3.0 Documentation

📋 Features:
• مدیریت کامل API Keys
• پردازش فایل‌های مختلف
• یادگیری Real-Time
• دیپ لرنینگ داخلی
• رابط کاربری فارسی
• API سرور RESTful
• سازگار با 32-bit

🚀 Getting Started:
1. ایجاد API Key جدید
2. آپلود فایل‌ها
3. آموزش مدل‌ها
4. استفاده از پیش‌بینی

🌐 API Endpoints:
• POST /register - ثبت اپلیکیشن
• POST /upload - آپلود فایل
• POST /learn - یادگیری
• POST /predict - پیش‌بینی
• GET /files - لیست فایل‌ها
• GET /stats - آمار سیستم

✅ Version: 3.0.0"""
            )
        
        def show_about_dialog(self):
            QMessageBox.about(
                self, "درباره ImanCore",
                """🧠 ImanCore v3.0
هسته مرکزی هوش مصنوعی با دیپ لرنینگ داخلی

📊 Features:
✅ مدیریت کامل API Keyها
✅ پردازش هوشمند فایل‌ها
✅ یادگیری Real-Time + دیپ لرنینگ داخلی
✅ پیش‌بینی‌های هوشمند
✅ رابط کاربری گرافیکی فارسی
✅ API سرور RESTful
✅ سازگار با سیستم‌های 32-bit
✅ امنیت کامل
✅ دیپ لرنینگ داخلی (بدون وابستگی خارجی)

🛠️ Developed with:
• Python 3.8+
• PyQt5 for GUI
• FastAPI for API
• SQLite for storage
• Pure Python Deep Learning

📅 Version: 3.0.0
👨‍💻 Developer: ImanCore Team"""
            )
        
        def closeEvent(self, event):
            reply = QMessageBox.question(
                self, "خروج",
                "آیا مطمئن هستید که می‌خواهید برنامه را ببندید؟",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.core.shutdown()
                event.accept()
            else:
                event.ignore()
    
    def run_gui():
        """اجرای رابط کاربری"""
        print("🖥️ Starting ImanCore GUI...")
        app = QApplication(sys.argv)
        app.setApplicationName("ImanCore")
        app.setApplicationVersion("3.0.0")
        
        window = ImanCoreGUI()
        window.show()
        
        exit_code = app.exec_()
        
        print("🔴 ImanCore GUI shutdown complete")
        sys.exit(exit_code)

# ============================================================================
# تابع اصلی
# ============================================================================

def main():
    """تابع اصلی برنامه"""
    print("🚀 Starting ImanCore v3.0...")
    print("=" * 60)
    
    # بررسی حالت اجرا
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "api" and FASTAPI_AVAILABLE:
            # اجرای API سرور
            host = "0.0.0.0"
            port = 8000
            
            if len(sys.argv) > 2:
                host = sys.argv[2]
            if len(sys.argv) > 3:
                port = int(sys.argv[3])
            
            run_api_server(host, port)
            
        elif mode == "gui" and PYQT_AVAILABLE:
            # اجرای GUI
            run_gui()
            
        elif mode == "both":
            # اجرای هر دو (در تردهای جداگانه)
            if FASTAPI_AVAILABLE and PYQT_AVAILABLE:
                import threading
                
                # اجرای API سرور در ترد جداگانه
                api_thread = threading.Thread(
                    target=run_api_server,
                    args=("0.0.0.0", 8000),
                    daemon=True
                )
                api_thread.start()
                
                # اجرای GUI در ترد اصلی
                run_gui()
                
            else:
                print("❌ برای اجرای هر دو حالت نیاز به PyQt5 و FastAPI دارید")
                print("📦 نصب: pip install PyQt5 fastapi uvicorn")
        
        else:
            print(f"❌ حالت نامعتبر: {mode}")
            print("✅ حالت‌های مجاز: api, gui, both")
            print("📌 مثال: python imancore_v3.py api")
            print("📌 مثال: python imancore_v3.py gui")
            print("📌 مثال: python imancore_v3.py both")
    
    else:
        # حالت پیش‌فرض: GUI اگر موجود باشد، در غیر این صورت API
        if PYQT_AVAILABLE:
            run_gui()
        elif FASTAPI_AVAILABLE:
            run_api_server()
        else:
            print("❌ هیچ رابطی در دسترس نیست!")
            print("📦 نصب PyQt5: pip install PyQt5")
            print("📦 نصب FastAPI: pip install fastapi uvicorn")
            print("📌 یا از خط فرمان استفاده کنید:")
            print("   python imancore_v3.py api")
            print("   python imancore_v3.py gui")

# ============================================================================
# ورودی برنامه
# ============================================================================

if __name__ == "__main__":
    main()