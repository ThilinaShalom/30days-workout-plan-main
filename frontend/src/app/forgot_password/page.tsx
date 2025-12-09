'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';
import AppNavbar from '@/components/AppNavbar';
import { Container, Form, Button, Alert, Card } from 'react-bootstrap';

export default function ForgotPassword() {
    const [email, setEmail] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setSuccess('');
        setLoading(true);

        try {
            const response = await api.post('/auth/forgot-password', { email });
            setSuccess(response.data.message);

            // If in development, log the reset link
            if (response.data.resetLink) {
                console.log('Password Reset Link:', response.data.resetLink);
            }

            // Redirect to login after 3 seconds
            setTimeout(() => {
                router.push('/login');
            }, 3000);
        } catch (err: any) {
            setError(err.response?.data?.error || 'Failed to send reset email');
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <AppNavbar brand="Forgot Password" />
            <Container className="mt-5">
                <div className="row justify-content-center">
                    <div className="col-md-6">
                        <Card style={{ background: '#1a1a1a', color: '#e0e0e0', border: '1px solid #00ccff' }}>
                            <Card.Body>
                                <h2 className="text-center mb-4" style={{ color: '#00ccff' }}>
                                    Reset Password
                                </h2>

                                {error && <Alert variant="danger">{error}</Alert>}
                                {success && (
                                    <Alert variant="success">
                                        {success}
                                        <br />
                                        <small>Redirecting to login page...</small>
                                    </Alert>
                                )}

                                <Form onSubmit={handleSubmit}>
                                    <Form.Group className="mb-3">
                                        <Form.Label>Email Address</Form.Label>
                                        <Form.Control
                                            type="email"
                                            placeholder="Enter your email"
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            required
                                            style={{
                                                background: '#252525',
                                                color: '#fff',
                                                border: '1px solid #00ccff',
                                            }}
                                        />
                                        <Form.Text className="text-muted">
                                            Enter the email associated with your account
                                        </Form.Text>
                                    </Form.Group>

                                    <Button
                                        variant="primary"
                                        type="submit"
                                        className="w-100"
                                        disabled={loading}
                                        style={{
                                            background: 'linear-gradient(90deg, #0077ff, #00ccff)',
                                            border: 'none',
                                        }}
                                    >
                                        {loading ? 'Sending...' : 'Send Reset Link'}
                                    </Button>
                                </Form>

                                <div className="text-center mt-3">
                                    <a
                                        href="/login"
                                        style={{ color: '#00ccff', textDecoration: 'none' }}
                                    >
                                        ← Back to Login
                                    </a>
                                </div>
                            </Card.Body>
                        </Card>
                    </div>
                </div>
            </Container>
        </>
    );
}
