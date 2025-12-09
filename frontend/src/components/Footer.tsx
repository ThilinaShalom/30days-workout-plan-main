'use client';
import Link from 'next/link';

export default function Footer() {
    return (
        <footer style={{
            background: 'linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%)',
            borderTop: '1px solid rgba(0, 204, 255, 0.2)',
            position: 'relative',
            overflow: 'hidden'
        }}>
            {/* Gradient glow effect */}
            <div style={{
                position: 'absolute',
                top: 0,
                left: '50%',
                transform: 'translateX(-50%)',
                width: '600px',
                height: '2px',
                background: 'linear-gradient(90deg, transparent, #00ccff, transparent)',
                boxShadow: '0 0 20px rgba(0, 204, 255, 0.5)'
            }}></div>

            <div className="container py-5">
                <div className="row g-4">
                    {/* Brand Section */}
                    <div className="col-lg-4 col-md-6">
                        <div style={{ marginBottom: '1.5rem' }}>
                            <h3 style={{
                                fontSize: '1.8rem',
                                fontWeight: 'bold',
                                background: 'linear-gradient(135deg, #00ccff, #0077ff)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                marginBottom: '1rem',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px'
                            }}>
                                <i className="ph ph-barbell" style={{ fontSize: '2rem', color: '#00ccff' }}></i>
                                FitGen AI
                            </h3>
                            <p style={{
                                color: '#fff',
                                opacity: 0.8,
                                fontSize: '0.95rem',
                                lineHeight: '1.6',
                                marginBottom: '1.5rem'
                            }}>
                                Transform your fitness journey with AI-powered personalized workout and nutrition plans. Your path to peak performance starts here.
                            </p>
                            {/* Social Links */}
                            <div style={{ display: 'flex', gap: '12px' }}>
                                {['x-logo', 'facebook-logo', 'instagram-logo', 'linkedin-logo'].map((social, idx) => (
                                    <a
                                        key={idx}
                                        href="#"
                                        style={{
                                            width: '40px',
                                            height: '40px',
                                            borderRadius: '10px',
                                            background: 'rgba(0, 204, 255, 0.1)',
                                            border: '1px solid rgba(0, 204, 255, 0.3)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            color: '#00ccff',
                                            fontSize: '1.2rem',
                                            transition: 'all 0.3s ease',
                                            textDecoration: 'none'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.background = 'rgba(0, 204, 255, 0.2)';
                                            e.currentTarget.style.transform = 'translateY(-3px)';
                                            e.currentTarget.style.boxShadow = '0 5px 15px rgba(0, 204, 255, 0.3)';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.background = 'rgba(0, 204, 255, 0.1)';
                                            e.currentTarget.style.transform = 'translateY(0)';
                                            e.currentTarget.style.boxShadow = 'none';
                                        }}
                                    >
                                        <i className={`ph ph-${social}`}></i>
                                    </a>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Quick Links */}
                    <div className="col-lg-2 col-md-6">
                        <h4 style={{
                            fontSize: '1.1rem',
                            fontWeight: 'bold',
                            color: '#fff',
                            marginBottom: '1.5rem',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}>
                            Quick Links
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                            {[
                                { label: 'Home', href: '/' },
                                { label: 'About Us', href: '#' },
                                { label: 'Features', href: '#' },
                                { label: 'Pricing', href: '#' }
                            ].map((link, idx) => (
                                <li key={idx} style={{ marginBottom: '0.75rem' }}>
                                    <Link
                                        href={link.href}
                                        style={{
                                            color: '#fff',
                                            opacity: 0.7,
                                            textDecoration: 'none',
                                            fontSize: '0.95rem',
                                            transition: 'all 0.3s ease',
                                            display: 'inline-flex',
                                            alignItems: 'center',
                                            gap: '8px'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.opacity = '1';
                                            e.currentTarget.style.color = '#00ccff';
                                            e.currentTarget.style.paddingLeft = '8px';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.opacity = '0.7';
                                            e.currentTarget.style.color = '#fff';
                                            e.currentTarget.style.paddingLeft = '0';
                                        }}
                                    >
                                        <i className="ph ph-caret-right" style={{ fontSize: '0.8rem' }}></i>
                                        {link.label}
                                    </Link>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Services */}
                    <div className="col-lg-3 col-md-6">
                        <h4 style={{
                            fontSize: '1.1rem',
                            fontWeight: 'bold',
                            color: '#fff',
                            marginBottom: '1.5rem',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}>
                            Services
                        </h4>
                        <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                            {[
                                'AI Workout Plans',
                                'Nutrition Guidance',
                                'Progress Tracking',
                                'Coach Support'
                            ].map((service, idx) => (
                                <li key={idx} style={{ marginBottom: '0.75rem' }}>
                                    <a
                                        href="#"
                                        style={{
                                            color: '#fff',
                                            opacity: 0.7,
                                            textDecoration: 'none',
                                            fontSize: '0.95rem',
                                            transition: 'all 0.3s ease',
                                            display: 'inline-flex',
                                            alignItems: 'center',
                                            gap: '8px'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.opacity = '1';
                                            e.currentTarget.style.color = '#00ccff';
                                            e.currentTarget.style.paddingLeft = '8px';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.opacity = '0.7';
                                            e.currentTarget.style.color = '#fff';
                                            e.currentTarget.style.paddingLeft = '0';
                                        }}
                                    >
                                        <i className="ph ph-caret-right" style={{ fontSize: '0.8rem' }}></i>
                                        {service}
                                    </a>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Contact Info */}
                    <div className="col-lg-3 col-md-6">
                        <h4 style={{
                            fontSize: '1.1rem',
                            fontWeight: 'bold',
                            color: '#fff',
                            marginBottom: '1.5rem',
                            textTransform: 'uppercase',
                            letterSpacing: '1px'
                        }}>
                            Contact
                        </h4>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                                <i className="ph ph-envelope" style={{ color: '#00ccff', fontSize: '1.3rem', marginTop: '2px' }}></i>
                                <div>
                                    <p style={{ color: '#fff', opacity: 0.5, fontSize: '0.85rem', margin: 0, marginBottom: '4px' }}>Email</p>
                                    <a href="mailto:support@fitgen.ai" style={{ color: '#fff', opacity: 0.8, textDecoration: 'none', fontSize: '0.95rem' }}>
                                        support@fitgen.ai
                                    </a>
                                </div>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                                <i className="ph ph-phone" style={{ color: '#00ccff', fontSize: '1.3rem', marginTop: '2px' }}></i>
                                <div>
                                    <p style={{ color: '#fff', opacity: 0.5, fontSize: '0.85rem', margin: 0, marginBottom: '4px' }}>Phone</p>
                                    <a href="tel:+1234567890" style={{ color: '#fff', opacity: 0.8, textDecoration: 'none', fontSize: '0.95rem' }}>
                                        +1 (234) 567-890
                                    </a>
                                </div>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                                <i className="ph ph-map-pin" style={{ color: '#00ccff', fontSize: '1.3rem', marginTop: '2px' }}></i>
                                <div>
                                    <p style={{ color: '#fff', opacity: 0.5, fontSize: '0.85rem', margin: 0, marginBottom: '4px' }}>Location</p>
                                    <p style={{ color: '#fff', opacity: 0.8, fontSize: '0.95rem', margin: 0 }}>
                                        San Francisco, CA
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Bottom Bar */}
                <div style={{
                    marginTop: '3rem',
                    paddingTop: '2rem',
                    borderTop: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                    {/* Copyright - Centered */}
                    <div className="row mb-3">
                        <div className="col-12 text-center">
                            <p style={{ color: '#fff', opacity: 0.6, fontSize: '0.9rem', margin: 0 }}>
                                © {new Date().getFullYear()} FitGen AI. All rights reserved.
                            </p>
                        </div>
                    </div>

                    {/* Developer & Links */}
                    <div className="row align-items-center">
                        <div className="col-md-6 text-center text-md-start mb-3 mb-md-0">
                            <p style={{
                                color: '#fff',
                                opacity: 0.7,
                                fontSize: '0.85rem',
                                margin: 0,
                                display: 'flex',
                                alignItems: 'center',
                                gap: '6px',
                                justifyContent: 'start'
                            }}>
                                Developed by
                                <a
                                    href="https://nexgenxtm.com"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    style={{
                                        background: 'linear-gradient(135deg, #00ccff, #8b5cf6)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent',
                                        fontWeight: 'bold',
                                        fontSize: '0.9rem',
                                        textDecoration: 'none',
                                        transition: 'all 0.3s ease'
                                    }}
                                    onMouseEnter={(e) => {
                                        e.currentTarget.style.background = 'linear-gradient(135deg, #8b5cf6, #00ccff)';
                                        e.currentTarget.style.webkitBackgroundClip = 'text';
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.background = 'linear-gradient(135deg, #00ccff, #8b5cf6)';
                                        e.currentTarget.style.webkitBackgroundClip = 'text';
                                    }}
                                >
                                    NexgenX
                                </a>
                            </p>
                        </div>
                        <div className="col-md-6 text-center text-md-end">
                            <div style={{ display: 'inline-flex', gap: '1.5rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                                {['Privacy Policy', 'Terms of Service', 'Cookie Policy'].map((item, idx) => (
                                    <a
                                        key={idx}
                                        href="#"
                                        style={{
                                            color: '#fff',
                                            opacity: 0.6,
                                            textDecoration: 'none',
                                            fontSize: '0.9rem',
                                            transition: 'all 0.3s ease'
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.opacity = '1';
                                            e.currentTarget.style.color = '#00ccff';
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.opacity = '0.6';
                                            e.currentTarget.style.color = '#fff';
                                        }}
                                    >
                                        {item}
                                    </a>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Decorative bottom gradient */}
            <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: '3px',
                background: 'linear-gradient(90deg, #00ccff, #8b5cf6, #ef4444, #f59e0b, #10b981, #00ccff)',
                backgroundSize: '200% 100%',
                animation: 'gradient-shift 5s ease infinite'
            }}></div>

            <style jsx>{`
                @keyframes gradient-shift {
                    0%, 100% {
                        background-position: 0% 50%;
                    }
                    50% {
                        background-position: 100% 50%;
                    }
                }
            `}</style>
        </footer>
    );
}
