'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import AppNavbar from '@/components/AppNavbar';
import api from '@/lib/api';
import { Table, Button, Modal, Form, Badge } from 'react-bootstrap';
import { useToast } from '@/contexts/ToastContext';

export default function AdminDashboard() {
    const toast = useToast();
    const [coaches, setCoaches] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [showAddModal, setShowAddModal] = useState(false);
    const [showEditModal, setShowEditModal] = useState(false);
    const [selectedCoach, setSelectedCoach] = useState<any>(null);
    const [formData, setFormData] = useState({
        email: '',
        password: '',
        coach_name: '',
        specialization: '',
        profile_pic_url: '',
        services: [] as string[]
    });
    const router = useRouter();

    useEffect(() => {
        fetchCoaches();
    }, []);

    const fetchCoaches = async () => {
        try {
            const response = await api.get('/admin/dashboard');
            setCoaches(response.data.coaches);
        } catch (error) {
            console.error('Error fetching coaches', error);
            router.push('/admin/login');
        } finally {
            setLoading(false);
        }
    };

    const handleAddCoach = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            await api.post('/admin/register_coach', formData);
            toast.success('Coach registered successfully!');
            setShowAddModal(false);
            resetForm();
            fetchCoaches();
        } catch (error: any) {
            const errorMsg = error.response?.data?.error || 'Failed to register coach';
            toast.error(errorMsg);
            console.error('Error registering coach:', error);
        }
    };

    const handleDeleteCoach = async (email: string) => {
        if (!confirm('Are you sure you want to delete this coach?')) return;
        try {
            await api.post(`/admin/delete_coach/${email}`);
            toast.success('Coach deleted successfully!');
            fetchCoaches();
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Failed to delete coach');
        }
    };

    const handleEditCoach = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedCoach) return;
        try {
            const updateData = {
                coach_name: formData.coach_name,
                specialization: formData.specialization,
                profile_pic_url: formData.profile_pic_url,
                services: formData.services,
            };

            await api.post(`/admin/edit_coach/${selectedCoach.email}`, updateData);
            toast.success('Coach updated successfully!');
            setShowEditModal(false);
            resetForm();
            fetchCoaches();
        } catch (error: any) {
            const errorMsg = error.response?.data?.error || 'Failed to update coach';
            toast.error(errorMsg);
            console.error('Error updating coach:', error);
        }
    };

    const handleResetPassword = async (email: string) => {
        if (!confirm('Send password reset email to this coach?')) return;
        try {
            await api.post(`/admin/reset_coach_password/${email}`);
            toast.success('Password reset link generated!');
        } catch (error: any) {
            toast.error(error.response?.data?.error || 'Failed to send reset email');
        }
    };

    const openEditModal = (coach: any) => {
        setSelectedCoach(coach);
        setFormData({
            email: coach.email,
            password: '',
            coach_name: coach.username,
            specialization: coach.specialization,
            profile_pic_url: coach.profile_pic_url,
            services: coach.services || []
        });
        setShowEditModal(true);
    };

    const resetForm = () => {
        setFormData({
            email: '',
            password: '',
            coach_name: '',
            specialization: '',
            profile_pic_url: '',
            services: []
        });
        setSelectedCoach(null);
    };

    if (loading) return <div className="text-center mt-5"><div className="spinner-border text-primary" role="status"></div></div>;

    return (
        <>
            <AppNavbar brand="Admin Dashboard" />
            <div className="container mt-5">
                <div className="d-flex justify-content-between align-items-center mb-4">
                    <h2>Manage Coaches</h2>
                    <Button variant="primary" onClick={() => setShowAddModal(true)}>
                        <i className="ph ph-plus me-2"></i>Add Coach
                    </Button>
                </div>

                {coaches.length > 0 ? (
                    <Table striped bordered hover variant="dark" responsive>
                        <thead>
                            <tr>
                                <th>Profile</th>
                                <th>Name</th>
                                <th>Email</th>
                                <th>Specialization</th>
                                <th>Services</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {coaches.map((coach, idx) => (
                                <tr key={idx}>
                                    <td>
                                        {coach.profile_pic_url ? (
                                            <img
                                                src={coach.profile_pic_url}
                                                alt={coach.username}
                                                style={{
                                                    width: '50px',
                                                    height: '50px',
                                                    borderRadius: '50%',
                                                    objectFit: 'cover',
                                                    border: '2px solid #00ccff'
                                                }}
                                                onError={(e) => {
                                                    e.currentTarget.style.display = 'none';
                                                    e.currentTarget.nextElementSibling?.setAttribute('style', 'display: flex');
                                                }}
                                            />
                                        ) : null}
                                        <div
                                            style={{
                                                width: '50px',
                                                height: '50px',
                                                borderRadius: '50%',
                                                background: 'linear-gradient(135deg, #0077ff, #00ccff)',
                                                display: coach.profile_pic_url ? 'none' : 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                fontSize: '1.5rem',
                                                color: 'white',
                                                fontWeight: 'bold'
                                            }}
                                        >
                                            {coach.username?.charAt(0).toUpperCase() || '?'}
                                        </div>
                                    </td>
                                    <td>{coach.username}</td>
                                    <td>{coach.email}</td>
                                    <td>{coach.specialization}</td>
                                    <td>
                                        {coach.services?.map((service: string, i: number) => (
                                            <Badge key={i} bg="info" className="me-1">{service}</Badge>
                                        ))}
                                    </td>
                                    <td>
                                        <Button variant="warning" size="sm" className="me-2" onClick={() => openEditModal(coach)}>
                                            <i className="ph ph-pencil me-1"></i>Edit
                                        </Button>
                                        <Button variant="info" size="sm" className="me-2" onClick={() => handleResetPassword(coach.email)}>
                                            <i className="ph ph-key me-1"></i>Reset Password
                                        </Button>
                                        <Button variant="danger" size="sm" onClick={() => handleDeleteCoach(coach.email)}>
                                            <i className="ph ph-trash me-1"></i>Delete
                                        </Button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </Table>
                ) : (
                    <p>No coaches registered yet.</p>
                )}

                {/* Add Coach Modal */}
                <Modal show={showAddModal} onHide={() => { setShowAddModal(false); resetForm(); }} contentClassName="bg-dark text-white">
                    <Modal.Header closeButton closeVariant="white">
                        <Modal.Title>Add New Coach</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <Form onSubmit={handleAddCoach}>
                            <Form.Group className="mb-3">
                                <Form.Label>Coach Name</Form.Label>
                                <Form.Control type="text" value={formData.coach_name} onChange={(e) => setFormData({ ...formData, coach_name: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Email</Form.Label>
                                <Form.Control type="email" value={formData.email} onChange={(e) => setFormData({ ...formData, email: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Password</Form.Label>
                                <Form.Control type="password" value={formData.password} onChange={(e) => setFormData({ ...formData, password: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Specialization</Form.Label>
                                <Form.Control type="text" value={formData.specialization} onChange={(e) => setFormData({ ...formData, specialization: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Profile Picture URL</Form.Label>
                                <Form.Control type="url" value={formData.profile_pic_url} onChange={(e) => setFormData({ ...formData, profile_pic_url: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Services (comma separated)</Form.Label>
                                <Form.Control
                                    type="text"
                                    placeholder="e.g., Personal Training, Nutrition Coaching"
                                    onChange={(e) => setFormData({ ...formData, services: e.target.value.split(',').map(s => s.trim()) })}
                                />
                            </Form.Group>
                            <Button variant="primary" type="submit">Add Coach</Button>
                        </Form>
                    </Modal.Body>
                </Modal>

                {/* Edit Coach Modal */}
                <Modal show={showEditModal} onHide={() => { setShowEditModal(false); resetForm(); }} contentClassName="bg-dark text-white">
                    <Modal.Header closeButton closeVariant="white">
                        <Modal.Title>Edit Coach</Modal.Title>
                    </Modal.Header>
                    <Modal.Body>
                        <Form onSubmit={handleEditCoach}>
                            <Form.Group className="mb-3">
                                <Form.Label>Coach Name</Form.Label>
                                <Form.Control type="text" value={formData.coach_name} onChange={(e) => setFormData({ ...formData, coach_name: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Email</Form.Label>
                                <Form.Control type="email" value={formData.email} onChange={(e) => setFormData({ ...formData, email: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Specialization</Form.Label>
                                <Form.Control type="text" value={formData.specialization} onChange={(e) => setFormData({ ...formData, specialization: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Profile Picture URL</Form.Label>
                                <Form.Control type="url" value={formData.profile_pic_url} onChange={(e) => setFormData({ ...formData, profile_pic_url: e.target.value })} required />
                            </Form.Group>
                            <Form.Group className="mb-3">
                                <Form.Label>Services (comma separated)</Form.Label>
                                <Form.Control
                                    type="text"
                                    value={formData.services.join(', ')}
                                    onChange={(e) => setFormData({ ...formData, services: e.target.value.split(',').map(s => s.trim()) })}
                                />
                            </Form.Group>
                            <Button variant="primary" type="submit">Update Coach</Button>
                        </Form>
                    </Modal.Body>
                </Modal>
            </div>
        </>
    );
}
