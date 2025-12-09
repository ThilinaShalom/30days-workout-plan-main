'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import AppNavbar from '@/components/AppNavbar';
import api from '@/lib/api';
import { Modal, Button, Card, Badge, Row, Col } from 'react-bootstrap';

export default function CustomerDashboard() {
    const [userData, setUserData] = useState<any>(null);
    const [plans, setPlans] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedPlan, setSelectedPlan] = useState<any>(null);
    const [showModal, setShowModal] = useState(false);
    const router = useRouter();

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await api.get('/dashboard/customer');
                setUserData({ user_name: response.data.user_name });
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
        setShowModal(true);
    };

    const deletePlan = async (planId: string) => {
        if (!confirm("Are you sure you want to delete this plan?")) return;
        try {
            await api.post(`/plans/${planId}/delete`);
            setPlans(plans.filter(p => p.id !== planId));
        } catch (error) {
            alert('Failed to delete plan');
        }
    };

    const sendToCoach = async (planId: string) => {
        if (!confirm("Do you want to send this plan to a coach for review?")) return;
        try {
            const response = await api.post(`/plans/${planId}/send-to-coach`);
            if (response.data.success) {
                alert('Plan sent to coach successfully!');
                // Update local state
                setPlans(plans.map(p => p.id === planId ? { ...p, status: 'requested' } : p));
                if (selectedPlan && selectedPlan.id === planId) {
                    setSelectedPlan({ ...selectedPlan, status: 'requested' });
                }
            }
        } catch (error) {
            alert('Failed to send plan to coach');
        }
    };

    if (loading) return <div className="text-center mt-5"><div className="spinner-border text-primary" role="status"></div></div>;

    return (
        <>
            <AppNavbar brand="Customer Dashboard" />
            <div className="container" style={{
                marginTop: '80px',
                marginBottom: '40px',
                padding: 'clamp(15px, 3vw, 30px)'
            }}>
                <div style={{ marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: 'clamp(1.5rem, 4vw, 2rem)', marginBottom: '0.5rem' }}>
                        Welcome, {userData?.user_name}!
                    </h2>
                    <p style={{
                        color: 'rgba(255, 255, 255, 0.7)',
                        fontSize: 'clamp(0.875rem, 2vw, 1rem)',
                        marginBottom: '1rem'
                    }}>
                        Select an option below to get started:
                    </p>
                    <div className="d-flex flex-column flex-sm-row gap-3">
                        <Link href="/generate" className="btn btn-primary" style={{
                            minHeight: '44px',
                            width: '100%',
                            maxWidth: '300px'
                        }}>
                            <i className="ph ph-plus-circle me-2"></i>
                            Generate Fitness Plan
                        </Link>
                    </div>
                </div>

                <h3 style={{ fontSize: 'clamp(1.25rem, 3vw, 1.75rem)', marginBottom: '1rem' }}>
                    My Generated Plans
                </h3>
                <Row>
                    {plans.map((plan, index) => (
                        <Col md={4} key={plan.id} className="mb-4">
                            <Card style={{ background: 'rgba(0, 0, 0, 0.5)', border: '1px solid rgba(255, 255, 255, 0.1)' }}>
                                <Card.Body>
                                    <Card.Title style={{ color: '#00b8d4' }}>Plan {index + 1}</Card.Title>
                                    <Card.Text style={{ color: '#fff' }}>
                                        <strong>Fitness Goal:</strong> {
                                            plan.user_data?.exercise_type === 0 ? 'Weight Loss' :
                                                plan.user_data?.exercise_type === 1 ? 'Muscle Gain' :
                                                    plan.user_data?.exercise_type === 2 ? 'Endurance' :
                                                        plan.user_data?.exercise_type === 3 ? 'General Fitness' : 'Not specified'
                                        }
                                    </Card.Text>
                                    <Card.Text style={{ color: '#fff' }}>
                                        <strong>Status:</strong>{' '}
                                        <Badge bg={
                                            plan.status === 'approved' ? 'success' :
                                                plan.status === 'rejected' ? 'danger' :
                                                    plan.status === 'requested' ? 'warning' : 'secondary'
                                        }>
                                            {plan.status ? plan.status.replace('_', ' ') : 'Not Sent'}
                                        </Badge>
                                    </Card.Text>
                                    <Button variant="primary" onClick={() => handleShow(plan)} className="me-2">See Plan</Button>
                                    <Button variant="danger" onClick={() => deletePlan(plan.id)}>Delete Plan</Button>
                                </Card.Body>
                            </Card>
                        </Col>
                    ))}
                </Row>

                <Modal show={showModal} onHide={handleClose} size="xl" contentClassName="bg-dark text-white">
                    <Modal.Header closeButton closeVariant="white" style={{ borderBottom: '2px solid #00ccff', background: 'linear-gradient(135deg, rgba(0, 204, 255, 0.1), rgba(139, 92, 246, 0.1))' }}>
                        <Modal.Title style={{ color: '#00ccff', fontSize: '1.8rem', fontWeight: 'bold' }}>
                            <i className="ph ph-barbell me-2"></i>
                            Your Personalized Fitness Plan
                        </Modal.Title>
                    </Modal.Header>
                    <Modal.Body style={{ maxHeight: '70vh', overflowY: 'auto', padding: '30px' }}>
                        {selectedPlan && (
                            <>
                                {/* Plan Status Banner */}
                                <div className="mb-4 p-3" style={{
                                    background: selectedPlan.status === 'approved' ? 'rgba(16, 185, 129, 0.1)' :
                                        selectedPlan.status === 'rejected' ? 'rgba(239, 68, 68, 0.1)' :
                                            selectedPlan.status === 'requested' ? 'rgba(245, 158, 11, 0.1)' :
                                                'rgba(139, 92, 246, 0.1)',
                                    borderRadius: '12px',
                                    border: `1px solid ${selectedPlan.status === 'approved' ? '#10b981' :
                                        selectedPlan.status === 'rejected' ? '#ef4444' :
                                            selectedPlan.status === 'requested' ? '#f59e0b' : '#8b5cf6'
                                        }`
                                }}>
                                    <div className="d-flex align-items-center justify-content-between">
                                        <div>
                                            <strong style={{ fontSize: '1.1rem' }}>Status:</strong>{' '}
                                            <Badge bg={
                                                selectedPlan.status === 'approved' ? 'success' :
                                                    selectedPlan.status === 'rejected' ? 'danger' :
                                                        selectedPlan.status === 'requested' ? 'warning' : 'secondary'
                                            } style={{ fontSize: '1rem', padding: '8px 16px' }}>
                                                {selectedPlan.status ? selectedPlan.status.replace('_', ' ').toUpperCase() : 'NOT SENT'}
                                            </Badge>
                                        </div>
                                        {selectedPlan.cluster !== undefined && (
                                            <div style={{ color: '#00ccff' }}>
                                                <i className="ph ph-chart-line me-2"></i>
                                                <strong>Cluster:</strong> {selectedPlan.cluster}
                                            </div>
                                        )}
                                    </div>
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
                                                        height: '100%',
                                                        transition: 'all 0.3s ease'
                                                    }}
                                                        className="hover-glow">
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
                                                            {workout.type === 'Rest' && (
                                                                <p style={{ color: '#b3b3b3', fontSize: '0.9rem', fontStyle: 'italic' }}>
                                                                    Recovery day - Focus on stretching and light activity
                                                                </p>
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

                                {/* Coach Comment Section */}
                                {selectedPlan.coach_comment && (
                                    <div className={`p-4`} style={{
                                        background: selectedPlan.status === 'approved' ? 'rgba(16, 185, 129, 0.1)' :
                                            selectedPlan.status === 'rejected' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(0, 204, 255, 0.1)',
                                        borderRadius: '12px',
                                        border: `1px solid ${selectedPlan.status === 'approved' ? '#10b981' :
                                            selectedPlan.status === 'rejected' ? '#ef4444' : '#00ccff'
                                            }`
                                    }}>
                                        <h6 style={{
                                            color: selectedPlan.status === 'approved' ? '#10b981' :
                                                selectedPlan.status === 'rejected' ? '#ef4444' : '#00ccff',
                                            fontSize: '1.2rem',
                                            marginBottom: '12px'
                                        }}>
                                            <i className="ph ph-chats-circle me-2"></i>
                                            Coach's Feedback
                                        </h6>
                                        <p style={{ color: '#e0e0e0', fontSize: '1rem', marginBottom: 0, lineHeight: '1.6' }}>
                                            {selectedPlan.coach_comment}
                                        </p>
                                    </div>
                                )}
                            </>
                        )}
                    </Modal.Body>
                    <Modal.Footer style={{ borderTop: '2px solid rgba(0, 204, 255, 0.2)', padding: '20px' }}>
                        {selectedPlan && (!selectedPlan.status || ['new', 'not_sent'].includes(selectedPlan.status)) && (
                            <Button
                                variant="outline-warning"
                                onClick={() => sendToCoach(selectedPlan.id)}
                                style={{
                                    borderWidth: '2px',
                                    fontWeight: 'bold',
                                    padding: '10px 24px'
                                }}
                            >
                                <i className="ph ph-paper-plane-tilt me-2"></i>
                                Send to Coach for Review
                            </Button>
                        )}
                        {selectedPlan && selectedPlan.status === 'requested' && (
                            <Badge bg="warning" text="dark" className="p-3" style={{ fontSize: '1rem' }}>
                                <i className="ph ph-clock me-2"></i>
                                Waiting for coach review
                            </Badge>
                        )}
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
