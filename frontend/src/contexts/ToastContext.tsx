'use client';
import { createContext, useContext, useState, ReactNode } from 'react';
import { Toast, ToastContainer } from 'react-bootstrap';

interface ToastMessage {
    id: number;
    message: string;
    type: 'success' | 'error' | 'warning' | 'info';
}

interface ToastContextType {
    showToast: (message: string, type: 'success' | 'error' | 'warning' | 'info') => void;
    success: (message: string) => void;
    error: (message: string) => void;
    warning: (message: string) => void;
    info: (message: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToast = () => {
    const context = useContext(ToastContext);
    if (!context) {
        throw new Error('useToast must be used within ToastProvider');
    }
    return context;
};

export const ToastProvider = ({ children }: { children: ReactNode }) => {
    const [toasts, setToasts] = useState<ToastMessage[]>([]);

    const showToast = (message: string, type: 'success' | 'error' | 'warning' | 'info') => {
        const id = Date.now();
        setToasts((prev) => [...prev, { id, message, type }]);
        setTimeout(() => {
            setToasts((prev) => prev.filter((toast) => toast.id !== id));
        }, 4000);
    };

    const success = (message: string) => showToast(message, 'success');
    const error = (message: string) => showToast(message, 'error');
    const warning = (message: string) => showToast(message, 'warning');
    const info = (message: string) => showToast(message, 'info');

    const getIcon = (type: string) => {
        switch (type) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            case 'info': return 'ℹ️';
            default: return '📢';
        }
    };

    const getBackground = (type: string) => {
        switch (type) {
            case 'success': return 'linear-gradient(135deg, rgba(16, 185, 129, 0.95), rgba(5, 150, 105, 0.95))';
            case 'error': return 'linear-gradient(135deg, rgba(239, 68, 68, 0.95), rgba(220, 38, 38, 0.95))';
            case 'warning': return 'linear-gradient(135deg, rgba(245, 158, 11, 0.95), rgba(217, 119, 6, 0.95))';
            case 'info': return 'linear-gradient(135deg, rgba(0, 204, 255, 0.95), rgba(0, 153, 204, 0.95))';
            default: return 'linear-gradient(135deg, rgba(139, 92, 246, 0.95), rgba(109, 40, 217, 0.95))';
        }
    };

    return (
        <ToastContext.Provider value={{ showToast, success, error, warning, info }}>
            {children}
            <ToastContainer
                position="top-end"
                className="p-3"
                style={{ zIndex: 9999, position: 'fixed', top: '80px', right: '20px' }}
            >
                {toasts.map((toast) => (
                    <Toast
                        key={toast.id}
                        onClose={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
                        style={{
                            background: getBackground(toast.type),
                            border: 'none',
                            borderRadius: '12px',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
                            minWidth: '300px',
                            marginBottom: '10px',
                            backdropFilter: 'blur(10px)',
                        }}
                    >
                        <Toast.Header
                            closeVariant="white"
                            style={{
                                background: 'transparent',
                                border: 'none',
                                color: 'white',
                                fontWeight: 'bold',
                                fontSize: '1.1rem',
                            }}
                        >
                            <span className="me-2" style={{ fontSize: '1.5rem' }}>{getIcon(toast.type)}</span>
                            <strong className="me-auto" style={{ color: 'white' }}>
                                {toast.type.charAt(0).toUpperCase() + toast.type.slice(1)}
                            </strong>
                        </Toast.Header>
                        <Toast.Body style={{ color: 'white', fontSize: '1rem', padding: '12px 16px' }}>
                            {toast.message}
                        </Toast.Body>
                    </Toast>
                ))}
            </ToastContainer>
        </ToastContext.Provider>
    );
};
