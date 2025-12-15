import React, { useState } from 'react';
import emailjs from '@emailjs/browser';
import styles from './styles.module.css';

export default function ContactSection() {
    // Directly paste your keys here
    const emailjsServiceId = 'service_d2xoey4';
    const emailjsTemplateId = 'template_v15uzk7';
    const emailjsPublicKey = 'eoGOLaDdnkYYxPqnE';

    const [loading, setLoading] = useState(false);
    const [toast, setToast] = useState(false);

    const sendEmail = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setLoading(true);

        const form = e.currentTarget; // explicitly store form reference

        emailjs
            .sendForm(emailjsServiceId, emailjsTemplateId, form, emailjsPublicKey)
            .then(() => {
                setLoading(false);
                setToast(true);

                if (form) form.reset(); // reset only if form exists

                setTimeout(() => setToast(false), 4000);
            })
            .catch((error) => {
                setLoading(false);
                console.error('EmailJS Error:', error);
                alert('Failed to send message. Please try again later.');
            });
    };


    return (
        <section className={styles.contactSection}>
            <div className="container">
                <div className={styles.grid}>
                    <div className={styles.info}>
                        <h2>Contact Us</h2>
                        <p>
                            We are committed to processing your information and will contact
                            you soon regarding your AI & Robotics project.
                        </p>

                        <ul>
                            <li>📧 hamailsyed139@gmail.com</li>
                            <li>📍 Global · Remote · Open Source</li>
                            <li>📞 +92 336 3351905</li>
                        </ul>
                    </div>

                    <form className={styles.form} onSubmit={sendEmail}>
                        <input name="name" type="text" placeholder="Name *" required />
                        <input name="email" type="email" placeholder="Email *" required />
                        <textarea name="message" placeholder="Message *" required />

                        <button type="submit" disabled={loading}>
                            {loading ? 'Sending...' : 'Submit'}
                        </button>
                    </form>
                </div>
            </div>

            {toast && (
                <div className={styles.toast}>
                    ✨ We have received your message. We will contact you soon.
                </div>
            )}
        </section>
    );
}
