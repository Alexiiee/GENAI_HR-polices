css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem;
    display: flex
}

.chat-message.user {
    background-color: #2b313e
}

.chat-message.bot {
    background-color: #475063
}

.chat-message .avatar {
  width: 20%;
}

.chat-message .avatar img {
  max-width: 100px;
  max-height: 100px;
  border-radius: 50%;
  object-fit: cover;
}

.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_temp = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="./Code-files/ChatBot/bot.jpg" alt="Bot">
  </div>
  <div class="message">{{MSG}}</div>
</div>
'''

user_temp = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="./Code-files/ChatBot/bot.jpg" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''