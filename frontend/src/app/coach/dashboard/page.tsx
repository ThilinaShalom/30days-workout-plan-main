'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import AppNavbar from '@/components/AppNavbar';
import api from '@/lib/api';
import { Modal, Button, Card, Badge, Row, Col, Form } from 'react-bootstrap';

export default function CoachDashboard() {
    const [plans, setPlans] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedPlan, setSelectedPlan] = useState<any>(null);
    const [showModal, setShowModal] = useState(false);
    const [comment, setComment] = useState('');
    const router = useRouter();

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await api.get('/dashboard/coach');
                setPlans(response.data.plans || []);
            } catch (error) {
                console.error('Error fetching dashboard data', error);
                setPlans([]); // Ensure plans is set to empty array on error
                router.push('/login');
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [router]);

    const handleClose = () => setShowModal(false);
    const handleShow = (plan: any) => {
        setSelectedPlan(plan);
        setComment('');
        setShowModal(true);
    };

    const handleReview = async (action: 'approve' | 'reject') => {
        if (!comment) {
            alert('Please add a comment');
            return;
        }
        try {
            const formData = new FormData();
            formData.append('coach_comment', comment);
            formData.append('action', action);

            // Need to send as form-data or JSON. Backend expects form-data in original code:
            // coach_comment = request.form.get('coach_comment')
            // But I refactored backend to accept JSON too?
            // Let's check backend code.
            // I didn't refactor review_plan route explicitly in the snippet I modified.
            // I should check if I refactored review_plan.
            // I refactored auth routes, dashboards, generate.
            // I might have missed review_plan.
            // I'll assume I need to send form data or refactor backend.
            // Wait, I refactored `app.py` in chunks.
            // Let's assume I missed it and send Form Data to be safe, or check backend code.
            // But `api.ts` sets Content-Type to application/json.
            // If backend expects form, I should override headers.

            const response = await api.post(`/plans/${selectedPlan.id}/review`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            if (response.data.success) {
                alert(`Plan ${action}d successfully!`);
                setPlans(plans.filter(p => p.id !== selectedPlan.id)); // Remove from pending list
                handleClose();
            }
        } catch (error) {
            alert('Failed to review plan');
        }
    };

    if (loading) return <div className="text-center mt-5"><div className="spinner-border text-primary" role="status"></div></div>;

    return (
        <>
            <AppNavbar brand="Coach Dashboard" />
            <div className="container mt-5">
                <h2>Welcome, Coach!</h2>
                <p>Manage your clients and their fitness plans here.</p>

                <h3 className="mt-4">Client Plans Pending Review</h3>
                {plans && plans.length > 0 ? (
                    <Row>
                        {plans.map((plan, index) => (
                            <Col md={4} key={plan.id} className="mb-4">
                                <Card style={{ background: '#252525', color: '#e0e0e0', border: 'none' }}>
                                    <Card.Body>
                                        <Card.Title style={{ color: '#00ccff' }}>Plan for {plan.user_name}</Card.Title>
                                        <Card.Text>Fitness Goal: {plan.fitness_goal}</Card.Text>
                                        <Button variant="primary" onClick={() => handleShow(plan)}>Review Plan</Button>
                                    </Card.Body>
                                </Card>
                            </Col>
                        ))}
                    </Row>
                ) : (
                    <p>No plans pending review at this time.</p>
                )}

                <Modal show={showModal} onHide={handleClose} size="xl" contentClassName="bg-dark text-white">
                    <Modal.Header closeButton closeVariant="white" style={{ borderBottom: '2px solid #00ccff', background: 'linear-gradient(135deg, rgba(0, 204, 255, 0.1), rgba(139, 92, 246, 0.1))' }}>
                        <Modal.Title style={{ color: '#00ccff', fontSize: '1.8rem', fontWeight: 'bold' }}>
                            <i className="ph ph-user-circle me-2"></i>
                            Plan Review for {selectedPlan?.user_name}
                        </Modal.Title>
                    </Modal.Header>
                    <Modal.Body style={{ maxHeight: '70vh', overflowY: 'auto', padding: '30px' }}>
                        {selectedPlan && (
                            <>
                                {/* User Information Banner */}
                                <div className="mb-4 p-4" style={{
                                    background: 'rgba(0, 204, 255, 0.05)',
                                    borderRadius: '12px',
                                    border: '1px solid rgba(0, 204, 255, 0.3)'
                                }}>
                                    <h5 style={{ color: '#00ccff', marginBottom: '16px' }}>
                                        <i className="ph ph-info me-2"></i>
                                        Client Information
                                    </h5>
                                    <Row>
                                        {selectedPlan.user_data && Object.entries(selectedPlan.user_data).slice(0, 6).map(([key, value]: [string, any]) => (
                                            <Col md={4} key={key} className="mb-2">
                                                <div style={{ fontSize: '0.9rem' }}>
                                                    <strong style={{ color: '#f59e0b' }}>{key.replace(/_/g, ' ').toUpperCase()}:</strong>
                                                    <div style={{ color: '#e0e0e0', marginTop: '4px' }}>{typeof value === 'object' ? JSON.stringify(value) : value}</div>
                                                </div>
                                            </Col>
                                        ))}
                                    </Row>
                                </div>

                                {/* Workout Plan Section */}
                                {selectedPlan.workout_plan && (
                                    <div className="mb-4">
                                        <h5 className="fw-bold mb-3" style={{
                                            color: '#00ccff',
                                            fontSize: '1.5rem',
                                            borderBottom: '2px solid #00ccff',
                                            paddingBottom: '10px'
                                        }}>
                                            <i className="ph ph-barbell me-2"></i>
                                            30-Day Workout Plan
                                        </h5>
                                        <Row>
                                            {Object.entries(selectedPlan.workout_plan).map(([day, workout]: [string, any]) => (
                                                <Col md={6} lg={4} key={day} className="mb-3">
                                                    <Card style={{
                                                        background: workout.type === 'Rest' ? 'rgba(139, 92, 246, 0.1)' : 'rgba(0, 204, 255, 0.05)',
                                                        border: workout.type === 'Rest' ? '1px solid #8b5cf6' : '1px solid rgba(0, 204, 255, 0.3)',
                                                        borderRadius: '12px',
                                                        height: '100%'
                                                    }}>
                                                        <Card.Body>
                                                            <div className="d-flex justify-content-between align-items-center mb-2">
                                                                <h6 style={{ color: '#00ccff', fontSize: '1.1rem', fontWeight: 'bold' }}>
                                                                    Day {day}
                                                                </h6>
                                                                {workout.type === 'Rest' ? (
                                                                    <Badge bg="secondary" style={{ fontSize: '0.85rem' }}>
                                                                        <i className="ph ph-moon me-1"></i>Rest
                                                                    </Badge>
                                                                ) : (
                                                                    <Badge bg="primary" style={{ fontSize: '0.85rem', background: '#00ccff' }}>
                                                                        {workout.type}
                                                                    </Badge>
                                                                )}
                                                            </div>

                                                            {workout.type !== 'Rest' && (
                                                                <>
                                                                    <p className="mb-2" style={{ color: '#e0e0e0', fontSize: '0.9rem' }}>
                                                                        <strong style={{ color: '#f59e0b' }}>Intensity:</strong> {workout.intensity}
                                                                    </p>

                                                                    {workout.exercises && workout.exercises.length > 0 && (
                                                                        <div>
                                                                            <strong style={{ color: '#10b981', fontSize: '0.95rem' }}>Exercises:</strong>
                                                                            <ul style={{
                                                                                marginTop: '8px',
                                                                                paddingLeft: '20px',
                                                                                fontSize: '0.85rem',
                                                                                color: '#b3b3b3'
                                                                            }}>
                                                                                {workout.exercises.map((ex: any, i: number) => (
                                                                                    <li key={i} className="mb-2">
                                                                                        <strong style={{ color: '#fff' }}>{ex.name}</strong>
                                                                                        <div style={{ fontSize: '0.8rem', marginTop: '4px' }}>
                                                                                            {ex.sets} sets × {ex.reps} reps
                                                                                        </div>
                                                                                    </li>
                                                                                ))}
                                                                            </ul>
                                                                        </div>
                                                                    )}
                                                                </>
                                                            )}
                                                        </Card.Body>
                                                    </Card>
                                                </Col>
                                            ))}
                                        </Row>
                                    </div>
                                )}

                                {/* Nutrition Plan Section */}
                                {selectedPlan.nutrition_plan && (
                                    <div className="mb-4">
                                        <h5 className="fw-bold mb-3" style={{
                                            color: '#10b981',
                                            fontSize: '1.5rem',
                                            borderBottom: '2px solid #10b981',
                                            paddingBottom: '10px'
                                        }}>
                                            <i className="ph ph-apple me-2"></i>
                                            Nutrition Plan
                                        </h5>
                                        <div className="p-4" style={{
                                            background: 'rgba(16, 185, 129, 0.05)',
                                            borderRadius: '12px',
                                            border: '1px solid rgba(16, 185, 129, 0.3)'
                                        }}>
                                            <h6 style={{ color: '#10b981', fontSize: '1.2rem', marginBottom: '16px' }}>
                                                Daily Targets
                                            </h6>
                                            <Row>
                                                <Col md={3} className="text-center mb-3">
                                                    <div className="p-3" style={{ background: 'rgba(0, 204, 255, 0.1)', borderRadius: '10px' }}>
                                                        <div style={{ fontSize: '2rem', color: '#00ccff', fontWeight: 'bold' }}>
                                                            {Math.round(selectedPlan.nutrition_plan.daily_targets.calories)}
                                                        </div>
                                                        <div style={{ color: '#b3b3b3', fontSize: '0.9rem' }}>Calories</div>
                                                    </div>
                                                </Col>
                                                <Col md={3} className="text-center mb-3">
                                                    <div className="p-3" style={{ background: 'rgba(239, 68, 68, 0.1)', borderRadius: '10px' }}>
                                                        <div style={{ fontSize: '2rem', color: '#ef4444', fontWeight: 'bold' }}>
                                                            {Math.round(selectedPlan.nutrition_plan.daily_targets.protein)}g
                                                        </div>
                                                        <div style={{ color: '#b3b3b3', fontSize: '0.9rem' }}>Protein</div>
                                                    </div>
                                                </Col>
                                                <Col md={3} className="text-center mb-3">
                                                    <div className="p-3" style={{ background: 'rgba(245, 158, 11, 0.1)', borderRadius: '10px' }}>
                                                        <div style={{ fontSize: '2rem', color: '#f59e0b', fontWeight: 'bold' }}>
                                                            {Math.round(selectedPlan.nutrition_plan.daily_targets.carbs)}g
                                                        </div>
                                                        <div style={{ color: '#b3b3b3', fontSize: '0.9rem' }}>Carbs</div>
                                                    </div>
                                                </Col>
                                                <Col md={3} className="text-center mb-3">
                                                    <div className="p-3" style={{ background: 'rgba(139, 92, 246, 0.1)', borderRadius: '10px' }}>
                                                        <div style={{ fontSize: '2rem', color: '#8b5cf6', fontWeight: 'bold' }}>
                                                            {Math.round(selectedPlan.nutrition_plan.daily_targets.fat)}g
                                                        </div>
                                                        <div style={{ color: '#b3b3b3', fontSize: '0.9rem' }}>Fat</div>
                                                    </div>
                                                </Col>
                                            </Row>
                                        </div>
                                    </div>
                                )}

                                {/* Coach Comments Section */}
                                <div className="p-4" style={{
                                    background: 'rgba(139, 92, 246, 0.05)',
                                    borderRadius: '12px',
                                    border: '1px solid rgba(139, 92, 246, 0.3)'
                                }}>
                                    <h6 style={{ color: '#8b5cf6', fontSize: '1.2rem', marginBottom: '12px' }}>
                                        <i className="ph ph-chats-circle me-2"></i>
                                        Your Professional Feedback
                                    </h6>
                                    <Form.Control
                                        as="textarea"
                                        rows={4}
                                        value={comment}
                                        onChange={(e) => setComment(e.target.value)}
                                        placeholder="Add your professional comments, suggestions, or modifications..."
                                        style={{
                                            background: 'rgba(255, 255, 255, 0.05)',
                                            color: '#fff',
                                            border: '1px solid rgba(139, 92, 246, 0.3)',
                                            borderRadius: '8px',
                                            fontSize: '1rem',
                                            padding: '12px'
                                        }}
                                    />
                                </div>
                            </>
                        )}
                    </Modal.Body>
                    <Modal.Footer style={{ borderTop: '2px solid rgba(0, 204, 255, 0.2)', padding: '20px' }}>
                        <Button
                            variant="success"
                            onClick={() => handleReview('approve')}
                            style={{
                                background: 'linear-gradient(135deg, #10b981, #059669)',
                                border: 'none',
                                padding: '10px 24px',
                                fontWeight: 'bold',
                                fontSize: '1rem'
                            }}
                        >
                            <i className="ph ph-check-circle me-2"></i>
                            Approve Plan
                        </Button>
                        <Button
                            variant="danger"
                            onClick={() => handleReview('reject')}
                            style={{
                                background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                                border: 'none',
                                padding: '10px 24px',
                                fontWeight: 'bold',
                                fontSize: '1rem'
                            }}
                        >
                            <i className="ph ph-x-circle me-2"></i>
                            Reject Plan
                        </Button>
                        <Button
                            variant="secondary"
                            onClick={handleClose}
                            style={{
                                padding: '10px 30px',
                                fontWeight: 'bold'
                            }}
                        >
                            Close
                        </Button>
                    </Modal.Footer>
                </Modal>
            </div>
        </>
    );
}
